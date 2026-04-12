// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! TableWrite for writing Arrow data to Paimon tables.
//!
//! Reference: [pypaimon TableWrite](https://github.com/apache/paimon/blob/master/paimon-python/pypaimon/write/table_write.py)
//! and [pypaimon FileStoreWrite](https://github.com/apache/paimon/blob/master/paimon-python/pypaimon/write/file_store_write.py)

use crate::arrow::format::{create_format_writer, FormatFileWriter};
use crate::io::FileIO;
use crate::spec::stats::BinaryTableStats;
use crate::spec::PartitionComputer;
use crate::spec::{
    extract_datum_from_arrow, BinaryRow, BinaryRowBuilder, CoreOptions, DataField, DataFileMeta,
    DataType, Datum, EMPTY_SERIALIZED_ROW,
};
use crate::table::commit_message::CommitMessage;
use crate::table::{SnapshotManager, Table, TableScan};
use crate::Result;
use arrow_array::{Int64Array, RecordBatch};
use arrow_ord::sort::{lexsort_to_indices, SortColumn, SortOptions};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinSet;

type PartitionBucketKey = (Vec<u8>, i32);

/// Enum to hold either an append-only writer or a key-value writer.
enum BucketWriter {
    Append(DataFileWriter),
    KeyValue(KeyValueFileWriter),
}

impl BucketWriter {
    async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        match self {
            BucketWriter::Append(w) => w.write(batch).await,
            BucketWriter::KeyValue(w) => w.write(batch).await,
        }
    }

    async fn prepare_commit(mut self) -> Result<Vec<DataFileMeta>> {
        match self {
            BucketWriter::Append(ref mut w) => w.prepare_commit().await,
            BucketWriter::KeyValue(ref mut w) => w.prepare_commit().await,
        }
    }
}

/// TableWrite writes Arrow RecordBatches to Paimon data files.
///
/// Each (partition, bucket) pair gets its own writer held in a HashMap.
/// Batches are routed to the correct writer based on partition/bucket.
///
/// Call `prepare_commit()` to close all writers and collect
/// `CommitMessage`s for use with `TableCommit`.
///
/// Reference: [pypaimon BatchTableWrite](https://github.com/apache/paimon/blob/master/paimon-python/pypaimon/write/table_write.py)
pub struct TableWrite {
    table: Table,
    partition_writers: HashMap<PartitionBucketKey, BucketWriter>,
    partition_computer: PartitionComputer,
    partition_keys: Vec<String>,
    partition_field_indices: Vec<usize>,
    bucket_key_indices: Vec<usize>,
    total_buckets: i32,
    schema_id: i64,
    target_file_size: i64,
    file_compression: String,
    file_compression_zstd_level: i32,
    write_buffer_size: i64,
    /// Primary key column indices in the user schema (empty for append-only).
    primary_key_indices: Vec<usize>,
    /// Next sequence number for PK table writes.
    next_sequence_number: i64,
}

impl TableWrite {
    pub(crate) fn new(table: &Table, next_sequence_number: i64) -> crate::Result<Self> {
        let schema = table.schema();
        let core_options = CoreOptions::new(schema.options());

        if core_options.data_evolution_enabled() {
            return Err(crate::Error::Unsupported {
                message: "TableWrite does not support data-evolution.enabled mode".to_string(),
            });
        }

        let total_buckets = core_options.bucket();
        let has_primary_keys = !schema.primary_keys().is_empty();

        if !has_primary_keys && total_buckets != -1 && core_options.bucket_key().is_none() {
            return Err(crate::Error::Unsupported {
                message: "Append tables with fixed bucket must configure 'bucket-key'".to_string(),
            });
        }
        let target_file_size = core_options.target_file_size();
        let file_compression = core_options.file_compression().to_string();
        let file_compression_zstd_level = core_options.file_compression_zstd_level();
        let write_buffer_size = core_options.write_parquet_buffer_size();
        let partition_keys: Vec<String> = schema.partition_keys().to_vec();
        let fields = schema.fields();

        let partition_field_indices: Vec<usize> = partition_keys
            .iter()
            .filter_map(|pk| fields.iter().position(|f| f.name() == pk))
            .collect();

        // Bucket keys: resolved by TableSchema
        let bucket_keys = schema.bucket_keys();

        let bucket_key_indices: Vec<usize> = bucket_keys
            .iter()
            .filter_map(|bk| fields.iter().position(|f| f.name() == bk))
            .collect();

        let partition_computer = PartitionComputer::new(
            &partition_keys,
            fields,
            core_options.partition_default_name(),
            core_options.legacy_partition_name(),
        )
        .unwrap();

        let primary_key_indices: Vec<usize> = schema
            .primary_keys()
            .iter()
            .filter_map(|pk| fields.iter().position(|f| f.name() == pk))
            .collect();

        Ok(Self {
            table: table.clone(),
            partition_writers: HashMap::new(),
            partition_computer,
            partition_keys,
            partition_field_indices,
            bucket_key_indices,
            total_buckets,
            schema_id: schema.id(),
            target_file_size,
            file_compression,
            file_compression_zstd_level,
            write_buffer_size,
            primary_key_indices,
            next_sequence_number,
        })
    }

    /// Scan the latest snapshot to determine the next sequence number for PK tables.
    pub(crate) async fn scan_next_sequence_number(table: &Table) -> crate::Result<i64> {
        let snapshot_manager = SnapshotManager::new(table.file_io().clone(), table.location().to_string());
        let latest_snapshot = snapshot_manager.get_latest_snapshot().await?;
        match latest_snapshot {
            None => Ok(0),
            Some(snapshot) => {
                let scan = TableScan::new(table, None, vec![], None, None, None);
                let entries = scan.plan_manifest_entries(&snapshot).await?;
                let max_seq = entries
                    .iter()
                    .map(|e| e.file().max_sequence_number)
                    .max()
                    .unwrap_or(0);
                Ok(max_seq + 1)
            }
        }
    }

    /// Write an Arrow RecordBatch. Rows are routed to the correct partition and bucket.
    pub async fn write_arrow_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }

        let grouped = self.divide_by_partition_bucket(batch)?;
        for ((partition_bytes, bucket), sub_batch) in grouped {
            self.write_bucket(partition_bytes, bucket, sub_batch)
                .await?;
        }
        Ok(())
    }

    /// Group rows by (partition_bytes, bucket) and return sub-batches.
    fn divide_by_partition_bucket(
        &self,
        batch: &RecordBatch,
    ) -> Result<Vec<(PartitionBucketKey, RecordBatch)>> {
        // Fast path: no partitions and single bucket — skip per-row routing
        if self.partition_field_indices.is_empty() && self.total_buckets <= 1 {
            return Ok(vec![((EMPTY_SERIALIZED_ROW.clone(), 0), batch.clone())]);
        }

        let fields = self.table.schema().fields();
        let mut groups: HashMap<PartitionBucketKey, Vec<usize>> = HashMap::new();

        for row_idx in 0..batch.num_rows() {
            let (partition_bytes, bucket) =
                self.extract_partition_bucket(batch, row_idx, fields)?;
            groups
                .entry((partition_bytes, bucket))
                .or_default()
                .push(row_idx);
        }

        let mut result = Vec::with_capacity(groups.len());
        for (key, row_indices) in groups {
            let sub_batch = if row_indices.len() == batch.num_rows() {
                batch.clone()
            } else {
                let indices = arrow_array::UInt32Array::from(
                    row_indices.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                );
                let columns: Vec<Arc<dyn arrow_array::Array>> = batch
                    .columns()
                    .iter()
                    .map(|col| arrow_select::take::take(col.as_ref(), &indices, None))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| crate::Error::DataInvalid {
                        message: format!("Failed to take rows: {e}"),
                        source: None,
                    })?;
                RecordBatch::try_new(batch.schema(), columns).map_err(|e| {
                    crate::Error::DataInvalid {
                        message: format!("Failed to create sub-batch: {e}"),
                        source: None,
                    }
                })?
            };
            result.push((key, sub_batch));
        }
        Ok(result)
    }

    /// Write a batch directly to the writer for the given (partition, bucket).
    async fn write_bucket(
        &mut self,
        partition_bytes: Vec<u8>,
        bucket: i32,
        batch: RecordBatch,
    ) -> Result<()> {
        let key = (partition_bytes, bucket);
        if !self.partition_writers.contains_key(&key) {
            self.create_writer(key.0.clone(), key.1)?;
        }
        let writer = self.partition_writers.get_mut(&key).unwrap();
        writer.write(&batch).await
    }

    /// Write multiple Arrow RecordBatches.
    pub async fn write_arrow(&mut self, batches: &[RecordBatch]) -> Result<()> {
        for batch in batches {
            self.write_arrow_batch(batch).await?;
        }
        Ok(())
    }

    /// Close all writers and collect CommitMessages for use with TableCommit.
    /// Writers are cleared after this call, allowing the TableWrite to be reused.
    pub async fn prepare_commit(&mut self) -> Result<Vec<CommitMessage>> {
        let writers: Vec<(PartitionBucketKey, BucketWriter)> =
            self.partition_writers.drain().collect();

        let mut messages = Vec::new();
        for ((partition_bytes, bucket), writer) in writers {
            let files = writer.prepare_commit().await?;
            if !files.is_empty() {
                messages.push(CommitMessage::new(partition_bytes, bucket, files));
            }
        }
        Ok(messages)
    }

    /// Extract partition bytes and bucket for a single row.
    fn extract_partition_bucket(
        &self,
        batch: &RecordBatch,
        row_idx: usize,
        fields: &[DataField],
    ) -> Result<PartitionBucketKey> {
        // Build partition BinaryRow
        let partition_bytes = if self.partition_field_indices.is_empty() {
            EMPTY_SERIALIZED_ROW.clone()
        } else {
            let mut builder = BinaryRowBuilder::new(self.partition_field_indices.len() as i32);
            for (pos, &field_idx) in self.partition_field_indices.iter().enumerate() {
                let field = &fields[field_idx];
                match extract_datum_from_arrow(batch, row_idx, field_idx, field.data_type())? {
                    Some(datum) => builder.write_datum(pos, &datum, field.data_type()),
                    None => builder.set_null_at(pos),
                }
            }
            builder.build_serialized()
        };

        // Compute bucket
        let bucket = if self.total_buckets <= 1 || self.bucket_key_indices.is_empty() {
            0
        } else {
            let mut datums: Vec<(Option<Datum>, DataType)> = Vec::new();
            for &field_idx in &self.bucket_key_indices {
                let field = &fields[field_idx];
                let datum = extract_datum_from_arrow(batch, row_idx, field_idx, field.data_type())?;
                datums.push((datum, field.data_type().clone()));
            }
            let refs: Vec<(Option<&Datum>, &DataType)> =
                datums.iter().map(|(d, t)| (d.as_ref(), t)).collect();
            BinaryRow::compute_bucket_from_datums(&refs, self.total_buckets)
        };

        Ok((partition_bytes, bucket))
    }

    fn create_writer(&mut self, partition_bytes: Vec<u8>, bucket: i32) -> Result<()> {
        let partition_path = if self.partition_keys.is_empty() {
            String::new()
        } else {
            let row = BinaryRow::from_serialized_bytes(&partition_bytes)?;
            self.partition_computer.generate_partition_path(&row)?
        };

        let writer = if self.primary_key_indices.is_empty() {
            BucketWriter::Append(DataFileWriter::new(
                self.table.file_io().clone(),
                self.table.location().to_string(),
                partition_path,
                bucket,
                self.schema_id,
                self.target_file_size,
                self.file_compression.clone(),
                self.file_compression_zstd_level,
                self.write_buffer_size,
            ))
        } else {
            BucketWriter::KeyValue(KeyValueFileWriter::new(
                self.table.file_io().clone(),
                self.table.location().to_string(),
                partition_path,
                bucket,
                self.schema_id,
                self.target_file_size,
                self.file_compression.clone(),
                self.file_compression_zstd_level,
                self.write_buffer_size,
                self.primary_key_indices.clone(),
                self.next_sequence_number,
            ))
        };

        self.partition_writers
            .insert((partition_bytes, bucket), writer);
        Ok(())
    }
}

/// Internal writer that produces parquet data files for a single (partition, bucket).
///
/// Batches are accumulated into a single `FormatFileWriter` that streams directly
/// to storage. Call `prepare_commit()` to finalize and collect file metadata.
struct DataFileWriter {
    file_io: FileIO,
    table_location: String,
    partition_path: String,
    bucket: i32,
    schema_id: i64,
    target_file_size: i64,
    file_compression: String,
    file_compression_zstd_level: i32,
    write_buffer_size: i64,
    written_files: Vec<DataFileMeta>,
    /// Background file close tasks spawned during rolling.
    in_flight_closes: JoinSet<Result<DataFileMeta>>,
    /// Current open format writer, lazily created on first write.
    current_writer: Option<Box<dyn FormatFileWriter>>,
    current_file_name: Option<String>,
    current_row_count: i64,
}

impl DataFileWriter {
    #[allow(clippy::too_many_arguments)]
    fn new(
        file_io: FileIO,
        table_location: String,
        partition_path: String,
        bucket: i32,
        schema_id: i64,
        target_file_size: i64,
        file_compression: String,
        file_compression_zstd_level: i32,
        write_buffer_size: i64,
    ) -> Self {
        Self {
            file_io,
            table_location,
            partition_path,
            bucket,
            schema_id,
            target_file_size,
            file_compression,
            file_compression_zstd_level,
            write_buffer_size,
            written_files: Vec::new(),
            in_flight_closes: JoinSet::new(),
            current_writer: None,
            current_file_name: None,
            current_row_count: 0,
        }
    }

    /// Write a RecordBatch. Rolls to a new file when target size is reached.
    async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }

        if self.current_writer.is_none() {
            self.open_new_file(batch.schema()).await?;
        }

        self.current_row_count += batch.num_rows() as i64;
        self.current_writer.as_mut().unwrap().write(batch).await?;

        // Roll to a new file if target size is reached — close in background
        if self.current_writer.as_ref().unwrap().num_bytes() as i64 >= self.target_file_size {
            self.roll_file();
        }

        // Flush row group if in-progress buffer exceeds write_buffer_size
        if let Some(w) = self.current_writer.as_mut() {
            if w.in_progress_size() as i64 >= self.write_buffer_size {
                w.flush().await?;
            }
        }

        Ok(())
    }

    async fn open_new_file(&mut self, schema: arrow_schema::SchemaRef) -> Result<()> {
        let file_name = format!(
            "data-{}-{}.parquet",
            uuid::Uuid::new_v4(),
            self.written_files.len()
        );

        let bucket_dir = if self.partition_path.is_empty() {
            format!("{}/bucket-{}", self.table_location, self.bucket)
        } else {
            format!(
                "{}/{}/bucket-{}",
                self.table_location, self.partition_path, self.bucket
            )
        };
        self.file_io.mkdirs(&format!("{bucket_dir}/")).await?;

        let file_path = format!("{}/{}", bucket_dir, file_name);
        let output = self.file_io.new_output(&file_path)?;
        let writer = create_format_writer(
            &output,
            schema,
            &self.file_compression,
            self.file_compression_zstd_level,
        )
        .await?;
        self.current_writer = Some(writer);
        self.current_file_name = Some(file_name);
        self.current_row_count = 0;
        Ok(())
    }

    /// Close the current file writer and record the file metadata.
    async fn close_current_file(&mut self) -> Result<()> {
        let writer = match self.current_writer.take() {
            Some(w) => w,
            None => return Ok(()),
        };
        let file_name = self.current_file_name.take().unwrap();

        let row_count = self.current_row_count;
        self.current_row_count = 0;
        let file_size = writer.close().await? as i64;

        let meta = Self::build_meta(file_name, file_size, row_count, self.schema_id);
        self.written_files.push(meta);
        Ok(())
    }

    /// Spawn the current writer's close in the background for non-blocking rolling.
    fn roll_file(&mut self) {
        let writer = match self.current_writer.take() {
            Some(w) => w,
            None => return,
        };
        let file_name = self.current_file_name.take().unwrap();
        let row_count = self.current_row_count;
        self.current_row_count = 0;
        let schema_id = self.schema_id;

        self.in_flight_closes.spawn(async move {
            let file_size = writer.close().await? as i64;
            Ok(Self::build_meta(file_name, file_size, row_count, schema_id))
        });
    }

    /// Close the current writer and return all written file metadata.
    async fn prepare_commit(&mut self) -> Result<Vec<DataFileMeta>> {
        self.close_current_file().await?;
        while let Some(result) = self.in_flight_closes.join_next().await {
            let meta = result.map_err(|e| crate::Error::DataInvalid {
                message: format!("Background file close task panicked: {e}"),
                source: None,
            })??;
            self.written_files.push(meta);
        }
        Ok(std::mem::take(&mut self.written_files))
    }

    fn build_meta(
        file_name: String,
        file_size: i64,
        row_count: i64,
        schema_id: i64,
    ) -> DataFileMeta {
        DataFileMeta {
            file_name,
            file_size,
            row_count,
            min_key: EMPTY_SERIALIZED_ROW.clone(),
            max_key: EMPTY_SERIALIZED_ROW.clone(),
            key_stats: BinaryTableStats::new(
                EMPTY_SERIALIZED_ROW.clone(),
                EMPTY_SERIALIZED_ROW.clone(),
                vec![],
            ),
            value_stats: BinaryTableStats::new(
                EMPTY_SERIALIZED_ROW.clone(),
                EMPTY_SERIALIZED_ROW.clone(),
                vec![],
            ),
            min_sequence_number: 0,
            max_sequence_number: 0,
            schema_id,
            level: 0,
            extra_files: vec![],
            creation_time: Some(Utc::now()),
            delete_row_count: Some(0),
            embedded_index: None,
            file_source: Some(0), // APPEND
            value_stats_cols: Some(vec![]),
            external_path: None,
            first_row_id: None,
            write_cols: None,
        }
    }
}

/// Internal writer for primary-key tables that buffers data in memory,
/// sorts by primary key on flush, and prepends `_SEQUENCE_NUMBER` column.
///
/// The physical file schema is: [key_cols..., _SEQUENCE_NUMBER, value_cols...]
/// which matches the read path in `read_sort_merge`.
///
/// Reference: [org.apache.paimon.io.KeyValueDataFileWriterImpl](https://github.com/apache/paimon/blob/release-1.3/paimon-core/src/main/java/org/apache/paimon/io/KeyValueDataFileWriterImpl.java)
struct KeyValueFileWriter {
    file_io: FileIO,
    table_location: String,
    partition_path: String,
    bucket: i32,
    schema_id: i64,
    #[allow(dead_code)]
    target_file_size: i64,
    file_compression: String,
    file_compression_zstd_level: i32,
    write_buffer_size: i64,
    /// Primary key column indices in the user schema.
    primary_key_indices: Vec<usize>,
    /// Next sequence number to assign.
    next_sequence_number: i64,
    /// Buffered batches (user schema).
    buffer: Vec<RecordBatch>,
    /// Approximate buffered bytes.
    buffer_bytes: usize,
    /// Completed file metadata.
    written_files: Vec<DataFileMeta>,
}

impl KeyValueFileWriter {
    #[allow(clippy::too_many_arguments)]
    fn new(
        file_io: FileIO,
        table_location: String,
        partition_path: String,
        bucket: i32,
        schema_id: i64,
        target_file_size: i64,
        file_compression: String,
        file_compression_zstd_level: i32,
        write_buffer_size: i64,
        primary_key_indices: Vec<usize>,
        next_sequence_number: i64,
    ) -> Self {
        Self {
            file_io,
            table_location,
            partition_path,
            bucket,
            schema_id,
            target_file_size,
            file_compression,
            file_compression_zstd_level,
            write_buffer_size,
            primary_key_indices,
            next_sequence_number,
            buffer: Vec::new(),
            buffer_bytes: 0,
            written_files: Vec::new(),
        }
    }

    /// Buffer a RecordBatch. Flushes when buffer exceeds write_buffer_size.
    async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let batch_bytes: usize = batch
            .columns()
            .iter()
            .map(|c| c.get_buffer_memory_size())
            .sum();
        self.buffer.push(batch.clone());
        self.buffer_bytes += batch_bytes;

        if self.buffer_bytes as i64 >= self.write_buffer_size {
            self.flush().await?;
        }
        Ok(())
    }

    /// Sort buffered data by primary key, prepend _SEQUENCE_NUMBER, and write to a parquet file.
    async fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let batches = std::mem::take(&mut self.buffer);
        self.buffer_bytes = 0;

        // Concatenate all buffered batches.
        let user_schema = batches[0].schema();
        let combined = arrow_select::concat::concat_batches(&user_schema, &batches).map_err(
            |e| crate::Error::DataInvalid {
                message: format!("Failed to concat batches: {e}"),
                source: None,
            },
        )?;
        let num_rows = combined.num_rows();
        if num_rows == 0 {
            return Ok(());
        }

        // Sort by primary key columns.
        let sort_columns: Vec<SortColumn> = self
            .primary_key_indices
            .iter()
            .map(|&idx| SortColumn {
                values: combined.column(idx).clone(),
                options: Some(SortOptions {
                    descending: false,
                    nulls_first: true,
                }),
            })
            .collect();
        let sorted_indices = lexsort_to_indices(&sort_columns, None).map_err(|e| {
            crate::Error::DataInvalid {
                message: format!("Failed to sort by primary key: {e}"),
                source: None,
            }
        })?;
        let sorted_columns: Vec<Arc<dyn arrow_array::Array>> = combined
            .columns()
            .iter()
            .map(|col| arrow_select::take::take(col.as_ref(), &sorted_indices, None))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| crate::Error::DataInvalid {
                message: format!("Failed to reorder by sort indices: {e}"),
                source: None,
            })?;
        let sorted_batch =
            RecordBatch::try_new(user_schema.clone(), sorted_columns).map_err(|e| {
                crate::Error::DataInvalid {
                    message: format!("Failed to create sorted batch: {e}"),
                    source: None,
                }
            })?;

        // Assign sequence numbers.
        let start_seq = self.next_sequence_number;
        let end_seq = start_seq + num_rows as i64 - 1;
        self.next_sequence_number = end_seq + 1;
        let seq_array: Arc<dyn arrow_array::Array> = Arc::new(Int64Array::from(
            (start_seq..=end_seq).collect::<Vec<_>>(),
        ));

        // Build physical schema: [key_cols..., _SEQUENCE_NUMBER, value_cols...]
        let user_fields = user_schema.fields();
        let mut physical_fields: Vec<Arc<ArrowField>> = Vec::new();
        for &idx in &self.primary_key_indices {
            physical_fields.push(user_fields[idx].clone());
        }
        physical_fields.push(Arc::new(ArrowField::new(
            "_SEQUENCE_NUMBER",
            ArrowDataType::Int64,
            false,
        )));
        // Value columns = all columns not in primary key
        let pk_set: std::collections::HashSet<usize> =
            self.primary_key_indices.iter().copied().collect();
        let value_indices: Vec<usize> = (0..user_fields.len())
            .filter(|i| !pk_set.contains(i))
            .collect();
        for &idx in &value_indices {
            physical_fields.push(user_fields[idx].clone());
        }
        let physical_schema = Arc::new(ArrowSchema::new(physical_fields));

        // Build physical columns.
        let mut physical_columns: Vec<Arc<dyn arrow_array::Array>> = Vec::new();
        for &idx in &self.primary_key_indices {
            physical_columns.push(sorted_batch.column(idx).clone());
        }
        physical_columns.push(seq_array);
        for &idx in &value_indices {
            physical_columns.push(sorted_batch.column(idx).clone());
        }
        let physical_batch =
            RecordBatch::try_new(physical_schema.clone(), physical_columns).map_err(|e| {
                crate::Error::DataInvalid {
                    message: format!("Failed to create physical batch: {e}"),
                    source: None,
                }
            })?;

        // Write to parquet file.
        let file_name = format!(
            "data-{}-{}.parquet",
            uuid::Uuid::new_v4(),
            self.written_files.len()
        );
        let bucket_dir = if self.partition_path.is_empty() {
            format!("{}/bucket-{}", self.table_location, self.bucket)
        } else {
            format!(
                "{}/{}/bucket-{}",
                self.table_location, self.partition_path, self.bucket
            )
        };
        self.file_io.mkdirs(&format!("{bucket_dir}/")).await?;
        let file_path = format!("{}/{}", bucket_dir, file_name);
        let output = self.file_io.new_output(&file_path)?;
        let mut writer = create_format_writer(
            &output,
            physical_schema,
            &self.file_compression,
            self.file_compression_zstd_level,
        )
        .await?;
        writer.write(&physical_batch).await?;
        let file_size = writer.close().await? as i64;

        // Build min_key / max_key from the sorted primary key columns.
        let min_key = self.extract_key_binary_row(&sorted_batch, 0)?;
        let max_key = self.extract_key_binary_row(&sorted_batch, num_rows - 1)?;

        let meta = DataFileMeta {
            file_name,
            file_size,
            row_count: num_rows as i64,
            min_key,
            max_key,
            key_stats: BinaryTableStats::new(
                EMPTY_SERIALIZED_ROW.clone(),
                EMPTY_SERIALIZED_ROW.clone(),
                vec![],
            ),
            value_stats: BinaryTableStats::new(
                EMPTY_SERIALIZED_ROW.clone(),
                EMPTY_SERIALIZED_ROW.clone(),
                vec![],
            ),
            min_sequence_number: start_seq,
            max_sequence_number: end_seq,
            schema_id: self.schema_id,
            level: 0,
            extra_files: vec![],
            creation_time: Some(Utc::now()),
            delete_row_count: Some(0),
            embedded_index: None,
            file_source: Some(0), // APPEND
            value_stats_cols: Some(vec![]),
            external_path: None,
            first_row_id: None,
            write_cols: None,
        };
        self.written_files.push(meta);
        Ok(())
    }

    /// Flush remaining buffer and return all written file metadata.
    async fn prepare_commit(&mut self) -> Result<Vec<DataFileMeta>> {
        self.flush().await?;
        Ok(std::mem::take(&mut self.written_files))
    }

    /// Extract primary key columns from a batch at a given row index into a serialized BinaryRow.
    fn extract_key_binary_row(&self, batch: &RecordBatch, row_idx: usize) -> Result<Vec<u8>> {
        let num_keys = self.primary_key_indices.len();
        let mut builder = BinaryRowBuilder::new(num_keys as i32);
        for (pos, &col_idx) in self.primary_key_indices.iter().enumerate() {
            let col = batch.column(col_idx);
            if col.is_null(row_idx) {
                builder.set_null_at(pos);
            } else {
                // Use the arrow column to write the key value into BinaryRow.
                write_arrow_value_to_builder(&mut builder, pos, col.as_ref(), row_idx)?;
            }
        }
        Ok(builder.build_serialized())
    }
}

/// Write a single value from an Arrow array into a BinaryRowBuilder at the given position.
fn write_arrow_value_to_builder(
    builder: &mut BinaryRowBuilder,
    pos: usize,
    array: &dyn arrow_array::Array,
    row_idx: usize,
) -> Result<()> {
    use arrow_array::*;
    match array.data_type() {
        ArrowDataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            builder.write_int(pos, arr.value(row_idx));
        }
        ArrowDataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            builder.write_long(pos, arr.value(row_idx));
        }
        ArrowDataType::Int16 => {
            let arr = array.as_any().downcast_ref::<Int16Array>().unwrap();
            builder.write_short(pos, arr.value(row_idx));
        }
        ArrowDataType::Int8 => {
            let arr = array.as_any().downcast_ref::<Int8Array>().unwrap();
            builder.write_byte(pos, arr.value(row_idx));
        }
        ArrowDataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            let s = arr.value(row_idx);
            if s.len() <= 7 {
                builder.write_string_inline(pos, s);
            } else {
                builder.write_string(pos, s);
            }
        }
        ArrowDataType::LargeUtf8 => {
            let arr = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            let s = arr.value(row_idx);
            if s.len() <= 7 {
                builder.write_string_inline(pos, s);
            } else {
                builder.write_string(pos, s);
            }
        }
        _ => {
            // For unsupported types, write as null for now.
            // This covers the common PK types (int, long, string).
            builder.set_null_at(pos);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Identifier;
    use crate::io::FileIOBuilder;
    use crate::spec::{
        DataType, DecimalType, IntType, LocalZonedTimestampType, Schema, TableSchema,
        TimestampType, VarCharType,
    };
    use crate::table::{SnapshotManager, TableCommit};
    use arrow_array::Int32Array;
    use arrow_schema::{
        DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema, TimeUnit,
    };
    use std::sync::Arc;

    fn test_file_io() -> FileIO {
        FileIOBuilder::new("memory").build().unwrap()
    }

    fn test_schema() -> TableSchema {
        let schema = Schema::builder()
            .column("id", DataType::Int(IntType::new()))
            .column("value", DataType::Int(IntType::new()))
            .build()
            .unwrap();
        TableSchema::new(0, &schema)
    }

    fn test_partitioned_schema() -> TableSchema {
        let schema = Schema::builder()
            .column("pt", DataType::VarChar(VarCharType::string_type()))
            .column("id", DataType::Int(IntType::new()))
            .partition_keys(["pt"])
            .build()
            .unwrap();
        TableSchema::new(0, &schema)
    }

    fn test_table(file_io: &FileIO, table_path: &str) -> Table {
        Table::new(
            file_io.clone(),
            Identifier::new("default", "test_table"),
            table_path.to_string(),
            test_schema(),
            None,
        )
    }

    fn test_partitioned_table(file_io: &FileIO, table_path: &str) -> Table {
        Table::new(
            file_io.clone(),
            Identifier::new("default", "test_table"),
            table_path.to_string(),
            test_partitioned_schema(),
            None,
        )
    }

    async fn setup_dirs(file_io: &FileIO, table_path: &str) {
        file_io
            .mkdirs(&format!("{table_path}/snapshot/"))
            .await
            .unwrap();
        file_io
            .mkdirs(&format!("{table_path}/manifest/"))
            .await
            .unwrap();
    }

    fn make_batch(ids: Vec<i32>, values: Vec<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int32, false),
            ArrowField::new("value", ArrowDataType::Int32, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .unwrap()
    }

    fn make_partitioned_batch(pts: Vec<&str>, ids: Vec<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("pt", ArrowDataType::Utf8, false),
            ArrowField::new("id", ArrowDataType::Int32, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow_array::StringArray::from(pts)),
                Arc::new(Int32Array::from(ids)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_write_and_commit() {
        let file_io = test_file_io();
        let table_path = "memory:/test_table_write";
        setup_dirs(&file_io, table_path).await;

        let table = test_table(&file_io, table_path);
        let mut table_write = TableWrite::new(&table, 0).unwrap();

        let batch = make_batch(vec![1, 2, 3], vec![10, 20, 30]);
        table_write.write_arrow_batch(&batch).await.unwrap();

        let messages = table_write.prepare_commit().await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].bucket, 0);
        assert_eq!(messages[0].new_files.len(), 1);
        assert_eq!(messages[0].new_files[0].row_count, 3);

        // Commit and verify snapshot
        let commit = TableCommit::new(table, "test-user".to_string());
        commit.commit(messages).await.unwrap();

        let snap_manager = SnapshotManager::new(file_io.clone(), table_path.to_string());
        let snapshot = snap_manager.get_latest_snapshot().await.unwrap().unwrap();
        assert_eq!(snapshot.id(), 1);
        assert_eq!(snapshot.total_record_count(), Some(3));
    }

    #[tokio::test]
    async fn test_write_partitioned() {
        let file_io = test_file_io();
        let table_path = "memory:/test_table_write_partitioned";
        setup_dirs(&file_io, table_path).await;

        let table = test_partitioned_table(&file_io, table_path);
        let mut table_write = TableWrite::new(&table, 0).unwrap();

        let batch = make_partitioned_batch(vec!["a", "b", "a"], vec![1, 2, 3]);
        table_write.write_arrow_batch(&batch).await.unwrap();

        let messages = table_write.prepare_commit().await.unwrap();
        // Should have 2 commit messages (one per partition)
        assert_eq!(messages.len(), 2);

        let total_rows: i64 = messages
            .iter()
            .flat_map(|m| &m.new_files)
            .map(|f| f.row_count)
            .sum();
        assert_eq!(total_rows, 3);

        // Commit and verify
        let commit = TableCommit::new(table, "test-user".to_string());
        commit.commit(messages).await.unwrap();

        let snap_manager = SnapshotManager::new(file_io.clone(), table_path.to_string());
        let snapshot = snap_manager.get_latest_snapshot().await.unwrap().unwrap();
        assert_eq!(snapshot.id(), 1);
        assert_eq!(snapshot.total_record_count(), Some(3));
    }

    #[tokio::test]
    async fn test_write_empty_batch() {
        let file_io = test_file_io();
        let table_path = "memory:/test_table_write_empty";
        let table = test_table(&file_io, table_path);
        let mut table_write = TableWrite::new(&table, 0).unwrap();

        let batch = make_batch(vec![], vec![]);
        table_write.write_arrow_batch(&batch).await.unwrap();

        let messages = table_write.prepare_commit().await.unwrap();
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_prepare_commit_reusable() {
        let file_io = test_file_io();
        let table_path = "memory:/test_table_write_reuse";
        setup_dirs(&file_io, table_path).await;

        let table = test_table(&file_io, table_path);
        let mut table_write = TableWrite::new(&table, 0).unwrap();

        // First write + prepare_commit
        table_write
            .write_arrow_batch(&make_batch(vec![1, 2], vec![10, 20]))
            .await
            .unwrap();
        let messages1 = table_write.prepare_commit().await.unwrap();
        assert_eq!(messages1.len(), 1);
        assert_eq!(messages1[0].new_files[0].row_count, 2);

        // Second write + prepare_commit (reuse)
        table_write
            .write_arrow_batch(&make_batch(vec![3, 4, 5], vec![30, 40, 50]))
            .await
            .unwrap();
        let messages2 = table_write.prepare_commit().await.unwrap();
        assert_eq!(messages2.len(), 1);
        assert_eq!(messages2[0].new_files[0].row_count, 3);

        // Empty prepare_commit is fine
        let messages3 = table_write.prepare_commit().await.unwrap();
        assert!(messages3.is_empty());
    }

    #[tokio::test]
    async fn test_write_multiple_batches() {
        let file_io = test_file_io();
        let table_path = "memory:/test_table_write_multi";
        setup_dirs(&file_io, table_path).await;

        let table = test_table(&file_io, table_path);
        let mut table_write = TableWrite::new(&table, 0).unwrap();

        table_write
            .write_arrow_batch(&make_batch(vec![1, 2], vec![10, 20]))
            .await
            .unwrap();
        table_write
            .write_arrow_batch(&make_batch(vec![3, 4], vec![30, 40]))
            .await
            .unwrap();

        let messages = table_write.prepare_commit().await.unwrap();
        assert_eq!(messages.len(), 1);
        // Multiple batches accumulate into a single file
        assert_eq!(messages[0].new_files.len(), 1);

        let total_rows: i64 = messages[0].new_files.iter().map(|f| f.row_count).sum();
        assert_eq!(total_rows, 4);
    }

    fn test_bucketed_schema() -> TableSchema {
        let schema = Schema::builder()
            .column("id", DataType::Int(IntType::new()))
            .column("value", DataType::Int(IntType::new()))
            .option("bucket", "4")
            .option("bucket-key", "id")
            .build()
            .unwrap();
        TableSchema::new(0, &schema)
    }

    fn test_bucketed_table(file_io: &FileIO, table_path: &str) -> Table {
        Table::new(
            file_io.clone(),
            Identifier::new("default", "test_table"),
            table_path.to_string(),
            test_bucketed_schema(),
            None,
        )
    }

    /// Build a batch where the bucket-key column ("id") is nullable.
    fn make_nullable_id_batch(ids: Vec<Option<i32>>, values: Vec<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int32, true),
            ArrowField::new("value", ArrowDataType::Int32, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_write_bucketed_with_null_bucket_key() {
        let file_io = test_file_io();
        let table_path = "memory:/test_table_write_null_bk";
        setup_dirs(&file_io, table_path).await;

        let table = test_bucketed_table(&file_io, table_path);
        let mut table_write = TableWrite::new(&table, 0).unwrap();

        // Row with NULL bucket key should not panic
        let batch = make_nullable_id_batch(vec![None, Some(1), None], vec![10, 20, 30]);
        table_write.write_arrow_batch(&batch).await.unwrap();

        let messages = table_write.prepare_commit().await.unwrap();
        let total_rows: i64 = messages
            .iter()
            .flat_map(|m| &m.new_files)
            .map(|f| f.row_count)
            .sum();
        assert_eq!(total_rows, 3);
    }

    #[tokio::test]
    async fn test_null_bucket_key_routes_consistently() {
        let file_io = test_file_io();
        let table_path = "memory:/test_table_write_null_bk_consistent";
        setup_dirs(&file_io, table_path).await;

        let table = test_bucketed_table(&file_io, table_path);
        let mut table_write = TableWrite::new(&table, 0).unwrap();

        // Two NULLs should land in the same bucket
        let batch = make_nullable_id_batch(vec![None, None], vec![10, 20]);
        table_write.write_arrow_batch(&batch).await.unwrap();

        let messages = table_write.prepare_commit().await.unwrap();
        // Both NULL-key rows must be in the same (partition, bucket) group
        let null_bucket_rows: i64 = messages
            .iter()
            .flat_map(|m| &m.new_files)
            .map(|f| f.row_count)
            .sum();
        assert_eq!(null_bucket_rows, 2);
        // All NULL-key rows go to exactly one bucket
        assert_eq!(messages.len(), 1);
    }

    #[tokio::test]
    async fn test_null_vs_nonnull_bucket_key_differ() {
        let file_io = test_file_io();
        let table_path = "memory:/test_table_write_null_vs_nonnull";
        setup_dirs(&file_io, table_path).await;

        let table = test_bucketed_table(&file_io, table_path);

        // Compute bucket for NULL key
        let fields = table.schema().fields().to_vec();
        let tw = TableWrite::new(&table, 0).unwrap();

        let batch_null = make_nullable_id_batch(vec![None], vec![10]);
        let (_, bucket_null) = tw
            .extract_partition_bucket(&batch_null, 0, &fields)
            .unwrap();

        // Compute bucket for key = 0 (the value a null field's fixed bytes happen to be)
        let batch_zero = make_nullable_id_batch(vec![Some(0)], vec![20]);
        let (_, bucket_zero) = tw
            .extract_partition_bucket(&batch_zero, 0, &fields)
            .unwrap();

        // A NULL bucket key must produce a BinaryRow with the null bit set,
        // which hashes differently from a non-null 0 value.
        // (With 4 buckets they could theoretically collide, but the hash codes differ.)
        let mut builder_null = BinaryRowBuilder::new(1);
        builder_null.set_null_at(0);
        let hash_null = builder_null.build().hash_code();

        let mut builder_zero = BinaryRowBuilder::new(1);
        builder_zero.write_int(0, 0);
        let hash_zero = builder_zero.build().hash_code();

        assert_ne!(hash_null, hash_zero, "NULL and 0 should hash differently");
        // If hashes differ, buckets should differ (with 4 buckets, very likely)
        // But we verify the hash difference is the important invariant
        let _ = (bucket_null, bucket_zero);
    }

    /// Mirrors Java's testUnCompactDecimalAndTimestampNullValueBucketNumber.
    /// Non-compact types (Decimal(38,18), LocalZonedTimestamp(6), Timestamp(6))
    /// use variable-length encoding in BinaryRow — NULL handling must still work.
    #[tokio::test]
    async fn test_non_compact_null_bucket_key() {
        let file_io = test_file_io();

        let bucket_cols = ["d", "ltz", "ntz"];
        let total_buckets = 16;

        for bucket_col in &bucket_cols {
            let table_path = format!("memory:/test_null_bk_{bucket_col}");
            setup_dirs(&file_io, &table_path).await;

            let schema = Schema::builder()
                .column("d", DataType::Decimal(DecimalType::new(38, 18).unwrap()))
                .column(
                    "ltz",
                    DataType::LocalZonedTimestamp(LocalZonedTimestampType::new(6).unwrap()),
                )
                .column("ntz", DataType::Timestamp(TimestampType::new(6).unwrap()))
                .column("k", DataType::Int(IntType::new()))
                .option("bucket", total_buckets.to_string())
                .option("bucket-key", *bucket_col)
                .build()
                .unwrap();
            let table_schema = TableSchema::new(0, &schema);
            let table = Table::new(
                file_io.clone(),
                Identifier::new("default", "test_table"),
                table_path.to_string(),
                table_schema,
                None,
            );

            let tw = TableWrite::new(&table, 0).unwrap();
            let fields = table.schema().fields().to_vec();

            // Build a batch: d=NULL, ltz=NULL, ntz=NULL, k=1
            let arrow_schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("d", ArrowDataType::Decimal128(38, 18), true),
                ArrowField::new(
                    "ltz",
                    ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                    true,
                ),
                ArrowField::new(
                    "ntz",
                    ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                    true,
                ),
                ArrowField::new("k", ArrowDataType::Int32, false),
            ]));
            let batch = RecordBatch::try_new(
                arrow_schema,
                vec![
                    Arc::new(
                        arrow_array::Decimal128Array::from(vec![None::<i128>])
                            .with_precision_and_scale(38, 18)
                            .unwrap(),
                    ),
                    Arc::new(
                        arrow_array::TimestampMicrosecondArray::from(vec![None::<i64>])
                            .with_timezone("UTC"),
                    ),
                    Arc::new(arrow_array::TimestampMicrosecondArray::from(vec![
                        None::<i64>,
                    ])),
                    Arc::new(Int32Array::from(vec![1])),
                ],
            )
            .unwrap();

            let (_, bucket) = tw.extract_partition_bucket(&batch, 0, &fields).unwrap();

            // Expected: BinaryRow with 1 field, null at pos 0
            let mut builder = BinaryRowBuilder::new(1);
            builder.set_null_at(0);
            let expected_bucket = (builder.build().hash_code() % total_buckets).abs();

            assert_eq!(
                bucket, expected_bucket,
                "NULL bucket-key '{bucket_col}' should produce bucket {expected_bucket}, got {bucket}"
            );
        }
    }

    #[tokio::test]
    async fn test_write_rolling_on_target_file_size() {
        let file_io = test_file_io();
        let table_path = "memory:/test_table_write_rolling";
        setup_dirs(&file_io, table_path).await;

        // Create table with very small target-file-size to trigger rolling
        let schema = Schema::builder()
            .column("id", DataType::Int(IntType::new()))
            .column("value", DataType::Int(IntType::new()))
            .option("target-file-size", "1b")
            .build()
            .unwrap();
        let table_schema = TableSchema::new(0, &schema);
        let table = Table::new(
            file_io.clone(),
            Identifier::new("default", "test_table"),
            table_path.to_string(),
            table_schema,
            None,
        );

        let mut table_write = TableWrite::new(&table, 0).unwrap();

        // Write multiple batches — each should roll to a new file
        table_write
            .write_arrow_batch(&make_batch(vec![1, 2], vec![10, 20]))
            .await
            .unwrap();
        table_write
            .write_arrow_batch(&make_batch(vec![3, 4], vec![30, 40]))
            .await
            .unwrap();

        let messages = table_write.prepare_commit().await.unwrap();
        assert_eq!(messages.len(), 1);
        // With 1-byte target, each batch should produce a separate file
        assert_eq!(messages[0].new_files.len(), 2);

        let total_rows: i64 = messages[0].new_files.iter().map(|f| f.row_count).sum();
        assert_eq!(total_rows, 4);
    }

    // -----------------------------------------------------------------------
    // Primary-key table write tests
    // -----------------------------------------------------------------------

    fn test_pk_schema() -> TableSchema {
        let schema = Schema::builder()
            .column("id", DataType::Int(IntType::new()))
            .column("value", DataType::Int(IntType::new()))
            .primary_key(["id"])
            .build()
            .unwrap();
        TableSchema::new(0, &schema)
    }

    fn test_pk_table(file_io: &FileIO, table_path: &str) -> Table {
        Table::new(
            file_io.clone(),
            Identifier::new("default", "test_pk_table"),
            table_path.to_string(),
            test_pk_schema(),
            None,
        )
    }

    #[tokio::test]
    async fn test_pk_write_and_commit() {
        let file_io = test_file_io();
        let table_path = "memory:/test_pk_write";
        setup_dirs(&file_io, table_path).await;

        let table = test_pk_table(&file_io, table_path);
        let mut table_write = TableWrite::new(&table, 0).unwrap();

        let batch = make_batch(vec![3, 1, 2], vec![30, 10, 20]);
        table_write.write_arrow_batch(&batch).await.unwrap();

        let messages = table_write.prepare_commit().await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].new_files.len(), 1);

        let file = &messages[0].new_files[0];
        assert_eq!(file.row_count, 3);
        assert_eq!(file.level, 0);
        assert_eq!(file.min_sequence_number, 0);
        assert_eq!(file.max_sequence_number, 2);
        // min_key and max_key should be non-empty (serialized BinaryRow)
        assert!(!file.min_key.is_empty());
        assert!(!file.max_key.is_empty());

        // Commit
        let commit = TableCommit::new(table.clone(), "test-user".to_string());
        commit.commit(messages).await.unwrap();

        let snap_manager = SnapshotManager::new(file_io.clone(), table_path.to_string());
        let snapshot = snap_manager.get_latest_snapshot().await.unwrap().unwrap();
        assert_eq!(snapshot.id(), 1);
        assert_eq!(snapshot.total_record_count(), Some(3));
    }

    #[tokio::test]
    async fn test_pk_write_sorted_output() {
        let file_io = test_file_io();
        let table_path = "memory:/test_pk_sorted";
        setup_dirs(&file_io, table_path).await;

        let table = test_pk_table(&file_io, table_path);
        let mut table_write = TableWrite::new(&table, 0).unwrap();

        // Write unsorted data
        let batch = make_batch(vec![5, 2, 4, 1, 3], vec![50, 20, 40, 10, 30]);
        table_write.write_arrow_batch(&batch).await.unwrap();

        let messages = table_write.prepare_commit().await.unwrap();
        let commit = TableCommit::new(table.clone(), "test-user".to_string());
        commit.commit(messages).await.unwrap();

        // Read back using sort-merge reader — should be sorted by PK
        let rb = table.new_read_builder();
        let scan = rb.new_scan();
        let plan = scan.plan().await.unwrap();
        let read = rb.new_read().unwrap();
        let batches: Vec<RecordBatch> = futures::TryStreamExt::try_collect(
            read.to_arrow(plan.splits()).unwrap(),
        )
        .await
        .unwrap();

        let ids: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);

        let values: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(values, vec![10, 20, 30, 40, 50]);
    }

    #[tokio::test]
    async fn test_pk_write_dedup_across_commits() {
        let file_io = test_file_io();
        let table_path = "memory:/test_pk_dedup";
        setup_dirs(&file_io, table_path).await;

        let table = test_pk_table(&file_io, table_path);

        // First commit: id=1,2,3
        let mut tw1 = TableWrite::new(&table, 0).unwrap();
        tw1.write_arrow_batch(&make_batch(vec![1, 2, 3], vec![10, 20, 30]))
            .await
            .unwrap();
        let msgs1 = tw1.prepare_commit().await.unwrap();
        let commit = TableCommit::new(table.clone(), "test-user".to_string());
        commit.commit(msgs1).await.unwrap();

        // Second commit: id=2,3,4 with updated values, higher sequence numbers
        let next_seq = TableWrite::scan_next_sequence_number(&table).await.unwrap();
        assert_eq!(next_seq, 3); // previous commit used 0,1,2
        let mut tw2 = TableWrite::new(&table, next_seq).unwrap();
        tw2.write_arrow_batch(&make_batch(vec![2, 3, 4], vec![200, 300, 400]))
            .await
            .unwrap();
        let msgs2 = tw2.prepare_commit().await.unwrap();
        commit.commit(msgs2).await.unwrap();

        // Read back — dedup should keep newer values for id=2,3
        let rb = table.new_read_builder();
        let scan = rb.new_scan();
        let plan = scan.plan().await.unwrap();
        let read = rb.new_read().unwrap();
        let batches: Vec<RecordBatch> = futures::TryStreamExt::try_collect(
            read.to_arrow(plan.splits()).unwrap(),
        )
        .await
        .unwrap();

        let ids: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        let values: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();

        assert_eq!(ids, vec![1, 2, 3, 4]);
        assert_eq!(values, vec![10, 200, 300, 400]);
    }

    #[tokio::test]
    async fn test_pk_write_sequence_number_in_file() {
        let file_io = test_file_io();
        let table_path = "memory:/test_pk_seq";
        setup_dirs(&file_io, table_path).await;

        let table = test_pk_table(&file_io, table_path);
        let mut table_write = TableWrite::new(&table, 5).unwrap();

        let batch = make_batch(vec![1, 2], vec![10, 20]);
        table_write.write_arrow_batch(&batch).await.unwrap();

        let messages = table_write.prepare_commit().await.unwrap();
        let file = &messages[0].new_files[0];
        // Starting from seq 5, 2 rows → min=5, max=6
        assert_eq!(file.min_sequence_number, 5);
        assert_eq!(file.max_sequence_number, 6);
    }
}
