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

//! Shared field declarations and native options for global-index maintenance and raw reads.

use crate::spec::CoreOptions;
use crate::table::global_index_types::normalize_global_index_type;
use crate::table::{GlobalIndexBuildBuilder, Table};
use crate::Result;
use std::collections::{HashMap, HashSet};

pub(super) struct Definition {
    pub(super) column: String,
    pub(super) index_type: String,
    pub(super) options: HashMap<String, String>,
}

impl Definition {
    pub(super) fn builder<'a>(&self, table: &'a Table) -> GlobalIndexBuildBuilder<'a> {
        let mut builder = table.new_global_index_build_builder();
        builder
            .with_index_column(&self.column)
            .with_index_type(&self.index_type)
            .with_options(self.options.clone());
        builder
    }
}

/// Ownership is shared by maintenance and both table modes' read paths.
pub(super) fn declarations(options: &HashMap<String, String>) -> Result<Vec<(String, String)>> {
    declarations_impl(options, true)
}

/// Read planning can encounter unresolved legacy schema properties. Ignore
/// blank/unmatched declarations, while still rejecting competing column owners.
pub(super) fn declarations_for_read(
    options: &HashMap<String, String>,
) -> Result<Vec<(String, String)>> {
    declarations_impl(options, false)
}

fn declarations_impl(
    options: &HashMap<String, String>,
    strict: bool,
) -> Result<Vec<(String, String)>> {
    let core = CoreOptions::new(options);
    for kind in core.global_index_types() {
        if strict && kind != "vector" && normalize_global_index_type(&kind).is_none() {
            return Err(crate::Error::ConfigInvalid {
                message: format!("Unsupported global index type: {kind}"),
            });
        }
    }
    let mut result = Vec::new();
    let mut owners = HashSet::new();
    for family in ["btree", "bitmap", "multivalue", "fm", "full-text", "vector"] {
        let columns = if family == "vector" {
            core.vector_index_columns()
        } else {
            core.global_index_columns(family)
        };
        let mut unique = HashSet::new();
        for column in columns {
            if column.is_empty() && !strict {
                continue;
            }
            if column.is_empty() || !unique.insert(column.clone()) {
                return Err(crate::Error::ConfigInvalid { message: format!("global-index.{family}.index-columns (legacy pk-{family}.index.columns) contains an empty or duplicate column '{column}'") });
            }
            if !owners.insert(column.clone()) {
                return Err(crate::Error::ConfigInvalid { message: format!("Column '{column}' can own at most one primary-key index or automatically maintained global index") });
            }
            let kind = if family == "vector" && strict {
                vector_index_type(options, &column)?.ok_or_else(|| crate::Error::ConfigInvalid { message: format!("fields.{column}.global-index.vector.index-type must specify a supported vector index type") })?
            } else {
                family
            };
            result.push((column, kind.into()));
        }
    }
    Ok(result)
}

pub(super) fn definitions(table: &Table) -> Result<Vec<Definition>> {
    let options = table.schema().options();
    let owners = declarations(options)?
        .into_iter()
        .collect::<HashMap<_, _>>();
    for column in owners.keys() {
        if !table.schema().fields().iter().any(|f| f.name() == column) {
            return Err(crate::Error::ConfigInvalid {
                message: format!("Global index contains unknown column '{column}'"),
            });
        }
    }
    let mut result = Vec::with_capacity(owners.len());
    for field in table.schema().fields() {
        let Some(kind) = owners.get(field.name()) else {
            continue;
        };
        let family = if crate::vindex::is_vindex_index_type(kind) {
            "vector"
        } else {
            kind
        };
        let definition = Definition {
            column: field.name().into(),
            index_type: kind.clone(),
            options: field_index_options(options, field.name(), family, kind)?,
        };
        let mut merged = options.clone();
        merged.extend(definition.options.clone());
        crate::table::universal_compaction::UniversalCompaction::from_options(
            &merged,
            "global-index.",
        )?;
        result.push(definition);
    }
    Ok(result)
}

/// Strict native adapters receive resolved overrides without unrelated table properties.
pub(super) fn field_index_options(
    table_options: &HashMap<String, String>,
    column: &str,
    _family: &str,
    index_type: &str,
) -> Result<HashMap<String, String>> {
    let core = CoreOptions::new(table_options);
    let resolved = if crate::vindex::is_vindex_index_type(index_type) {
        core.vector_index_options(index_type, column, true)?
    } else {
        core.global_index_options(index_type, column)?
    };
    Ok(resolved
        .into_iter()
        .filter(|(key, value)| table_options.get(key) != Some(value))
        .collect())
}

pub(super) fn vector_index_type(
    options: &HashMap<String, String>,
    column: &str,
) -> Result<Option<&'static str>> {
    let Some(kind) = CoreOptions::new(options).vector_index_type(column)? else {
        return Ok(None);
    };
    normalize_global_index_type(&kind)
        .filter(|kind| crate::vindex::is_vindex_index_type(kind))
        .map(Some)
        .ok_or_else(|| crate::Error::ConfigInvalid {
            message: format!("Unsupported vector index type for {column}: {kind}"),
        })
}

pub(super) fn declares_column(
    options: &HashMap<String, String>,
    column: &str,
    family: &str,
) -> bool {
    let core = CoreOptions::new(options);
    let columns = if family == "vector" {
        core.vector_index_columns()
    } else {
        core.global_index_columns(family)
    };
    columns.iter().any(|c| c == column)
}

pub(super) fn declared_options(
    options: &HashMap<String, String>,
    column: &str,
    family: &str,
    index_type: &str,
) -> Result<HashMap<String, String>> {
    let prefixes = [
        format!("fields.{column}.global-index.{family}."),
        format!("fields.{column}.pk-{family}."),
    ];
    if declares_column(options, column, family)
        || options.contains_key(&format!("fields.{column}.index-options"))
        || options
            .keys()
            .any(|key| prefixes.iter().any(|prefix| key.starts_with(prefix)))
    {
        field_index_options(options, column, family, index_type)
    } else {
        Ok(HashMap::new())
    }
}
