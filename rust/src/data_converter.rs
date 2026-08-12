//! JSON -> minimal, bracket-free, quote-free text (and back again).

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

static LOOKS_NUMERIC: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$").expect("valid numeric regex"));

const RESERVED: [&str; 6] = ["true", "false", "null", "none", "~", "-"];

fn needs_quoting(text: &str) -> bool {
    if text.is_empty() || text.trim() != text {
        return true;
    }
    let lower = text.to_ascii_lowercase();
    if RESERVED.contains(&lower.as_str()) {
        return true;
    }
    if LOOKS_NUMERIC.is_match(text) {
        return true;
    }
    if text.contains('\n') || text.contains('\r') || text.contains('\t') {
        return true;
    }
    matches!(
        text.chars().next(),
        Some('"') | Some('\'') | Some('#') | Some('[') | Some('{') | Some('&')
            | Some('*') | Some('!') | Some('|') | Some('>') | Some('%') | Some('@') | Some('`')
    )
}

fn encode_scalar(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            if needs_quoting(s) {
                Value::String(s.clone()).to_string()
            } else {
                s.clone()
            }
        }
        other => other.to_string(),
    }
}

fn decode_scalar(token: &str) -> Value {
    if token.starts_with('"') {
        if let Ok(v) = serde_json::from_str::<Value>(token) {
            return v;
        }
    }
    match token {
        "null" => return Value::Null,
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        _ => {}
    }
    if LOOKS_NUMERIC.is_match(token) {
        if let Ok(v) = serde_json::from_str::<Value>(token) {
            return v;
        }
    }
    Value::String(token.to_string())
}

fn escape_key(key: &str) -> String {
    key.replace('\\', "\\\\").replace('.', "\\.")
}

fn unescape_key(key: &str) -> String {
    key.replace("\\.", ".").replace("\\\\", "\\")
}

fn split_path(path: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut buf = String::new();
    let mut chars = path.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() {
                buf.push(c);
                buf.push(n);
            }
            continue;
        }
        if c == '.' {
            parts.push(unescape_key(&buf));
            buf.clear();
            continue;
        }
        buf.push(c);
    }
    parts.push(unescape_key(&buf));
    parts
}

fn flatten(node: &Value, prefix: &str, sink: &mut Vec<(String, String)>) {
    match node {
        Value::Object(map) => {
            if map.is_empty() {
                sink.push((prefix.to_string(), "{}".to_string()));
                return;
            }
            for (k, v) in map {
                let child = escape_key(k);
                let path = if prefix.is_empty() { child } else { format!("{}.{}", prefix, child) };
                flatten(v, &path, sink);
            }
        }
        Value::Array(items) => {
            if items.is_empty() {
                sink.push((prefix.to_string(), "[]".to_string()));
                return;
            }
            let all_scalar = items.iter().all(|v| !matches!(v, Value::Object(_) | Value::Array(_)));
            if all_scalar {
                let joined = items.iter().map(encode_scalar).collect::<Vec<_>>().join(", ");
                sink.push((prefix.to_string(), joined));
                return;
            }
            for (i, v) in items.iter().enumerate() {
                let path = if prefix.is_empty() { i.to_string() } else { format!("{}.{}", prefix, i) };
                flatten(v, &path, sink);
            }
        }
        scalar => sink.push((prefix.to_string(), encode_scalar(scalar))),
    }
}

/// Flatten nested JSON into `path value` lines with no brackets and minimal quoting.
pub fn json_to_minimal_yaml(data: &Value) -> String {
    let mut sink = Vec::new();
    flatten(data, "", &mut sink);
    sink.iter()
        .map(|(p, v)| format!("{} {}", p, v).trim_end().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

fn split_top_level(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut in_str = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_str {
            if c == '\\' {
                buf.push(c);
                if let Some(n) = chars.next() {
                    buf.push(n);
                }
                continue;
            }
            if c == '"' {
                in_str = false;
            }
            buf.push(c);
        } else if c == '"' {
            in_str = true;
            buf.push(c);
        } else if c == ',' {
            out.push(buf.clone());
            buf.clear();
        } else {
            buf.push(c);
        }
    }
    out.push(buf);
    out
}

fn relist(node: Value) -> Value {
    match node {
        Value::Object(map) => {
            let converted: Map<String, Value> =
                map.into_iter().map(|(k, v)| (k, relist(v))).collect();
            let all_digits = !converted.is_empty() && converted.keys().all(|k| k.chars().all(|c| c.is_ascii_digit()));
            if all_digits {
                let mut keys: Vec<&String> = converted.keys().collect();
                keys.sort_by_key(|k| k.parse::<usize>().unwrap_or(usize::MAX));
                let expected: Vec<String> = (0..converted.len()).map(|i| i.to_string()).collect();
                if keys.iter().map(|k| k.as_str()).collect::<Vec<_>>()
                    == expected.iter().map(|k| k.as_str()).collect::<Vec<_>>()
                {
                    return Value::Array(expected.iter().map(|k| converted[k].clone()).collect());
                }
            }
            Value::Object(converted)
        }
        other => other,
    }
}

/// Inverse of [`json_to_minimal_yaml`]; proves the transform is lossless.
pub fn minimal_yaml_to_json(text: &str) -> Value {
    let mut root = Value::Object(Map::new());

    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let (path, raw_value) = match line.find(' ') {
            Some(idx) => (&line[..idx], line[idx + 1..].trim()),
            None => (line, ""),
        };
        let parts = split_path(path);

        let value = if raw_value == "{}" {
            Value::Object(Map::new())
        } else if raw_value == "[]" {
            Value::Array(Vec::new())
        } else if raw_value.contains(',') && !raw_value.starts_with('"') {
            Value::Array(split_top_level(raw_value).iter().map(|v| decode_scalar(v.trim())).collect())
        } else {
            decode_scalar(raw_value)
        };

        let mut cursor = &mut root;
        for (i, part) in parts.iter().enumerate() {
            let last = i == parts.len() - 1;
            let obj = cursor.as_object_mut().expect("object cursor");
            if last {
                obj.insert(part.clone(), value.clone());
            } else {
                cursor = obj.entry(part.clone()).or_insert_with(|| Value::Object(Map::new()));
            }
        }
    }

    relist(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn round_trip(v: Value) {
        let flat = json_to_minimal_yaml(&v);
        assert_eq!(minimal_yaml_to_json(&flat), v, "flat was:\n{}", flat);
    }

    #[test]
    fn simple() {
        round_trip(json!({"a": 1}));
    }

    #[test]
    fn deep_nesting() {
        round_trip(json!({"a": {"b": {"c": {"d": "deep"}}}}));
    }

    #[test]
    fn scalar_arrays() {
        round_trip(json!({"list": [1, 2, 3]}));
    }

    #[test]
    fn object_arrays() {
        round_trip(json!({"objs": [{"x": 1}, {"x": 2}]}));
    }

    #[test]
    fn empties() {
        round_trip(json!({"d": {}, "l": []}));
    }

    #[test]
    fn ambiguous_strings_stay_quoted() {
        let v = json!({"num_str": "123", "bool_str": "true", "version": "15.4"});
        let flat = json_to_minimal_yaml(&v);
        assert!(flat.contains("num_str \"123\""));
        round_trip(v);
    }

    #[test]
    fn unambiguous_strings_lose_quotes() {
        let flat = json_to_minimal_yaml(&json!({"name": "payment-api"}));
        assert_eq!(flat, "name payment-api");
    }

    #[test]
    fn no_bracket_noise() {
        let flat = json_to_minimal_yaml(&json!({"a": {"b": [1, 2]}}));
        assert!(!flat.contains('{'));
        assert!(!flat.contains('['));
    }

    #[test]
    fn keys_with_dots() {
        round_trip(json!({"key.with.dots": "value"}));
    }

    #[test]
    fn unicode_and_padding() {
        round_trip(json!({"u": "café → ☃", "p": "  padded  ", "e": ""}));
    }

    #[test]
    fn smaller_than_pretty_json() {
        let v = json!({"service": {"name": "api", "port": 8080, "tags": ["a", "b"]}});
        let flat = json_to_minimal_yaml(&v);
        let pretty = serde_json::to_string_pretty(&v).unwrap();
        assert!(flat.len() < pretty.len());
    }
}
