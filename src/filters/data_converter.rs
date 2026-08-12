//! Structural JSON → quote-free, brace-free YAML flattener.
//!
//! The converter owns a complete recursive-descent JSON parser (no serde)
//! so key order and numeric lexemes survive. Non-empty objects and arrays
//! are emitted as block YAML; quotes appear only when a plain scalar would
//! be ambiguous. Empty collections keep `{}` / `[]` so the rewrite stays
//! information-preserving.

use crate::pipeline::TokenFilter;

/// Lossless JSON-to-YAML rewriter.
#[derive(Debug, Clone, Default)]
pub struct DataConverter;

impl DataConverter {
    pub fn new() -> Self {
        Self
    }

    /// Parse `input` as JSON and emit quote-free YAML. Non-JSON input is
    /// returned unchanged so the filter is safe inside a mixed pipeline.
    pub fn convert(&self, input: &str) -> String {
        match parse_json(input) {
            Ok(value) => {
                let mut out = String::with_capacity(input.len());
                emit_yaml(&value, 0, true, &mut out);
                if input.ends_with('\n') && !out.ends_with('\n') {
                    out.push('\n');
                }
                out
            }
            Err(_) => input.to_string(),
        }
    }

    /// Inverse: parse the YAML dialect this converter emits and rebuild a
    /// compact JSON document. Used by the self-test harness to prove
    /// losslessness.
    pub fn yaml_to_json(&self, yaml: &str) -> Result<String, String> {
        let value = parse_yaml(yaml)?;
        let mut out = String::new();
        emit_json(&value, &mut out);
        Ok(out)
    }
}

impl TokenFilter for DataConverter {
    fn filter(&self, input: &str) -> String {
        self.convert(input)
    }

    fn name(&self) -> &'static str {
        "data_converter"
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub offset: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at byte {}", self.message, self.offset)
    }
}

struct Cursor<'a> {
    src: &'a str,
    bytes: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            i: 0,
        }
    }

    fn eof(&self) -> bool {
        self.i >= self.bytes.len()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.i).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.i += 1;
        Some(c)
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn err(&self, message: impl Into<String>) -> ParseError {
        ParseError {
            message: message.into(),
            offset: self.i,
        }
    }
}

pub fn parse_json(src: &str) -> Result<Json, ParseError> {
    let mut c = Cursor::new(src);
    c.skip_ws();
    let value = parse_value(&mut c)?;
    c.skip_ws();
    if !c.eof() {
        return Err(c.err("trailing junk after top-level JSON value"));
    }
    Ok(value)
}

fn parse_value(c: &mut Cursor<'_>) -> Result<Json, ParseError> {
    c.skip_ws();
    match c.peek() {
        Some(b'{') => parse_object(c),
        Some(b'[') => parse_array(c),
        Some(b'"') => Ok(Json::String(parse_string(c)?)),
        Some(b't') => parse_lit(c, b"true", Json::Bool(true)),
        Some(b'f') => parse_lit(c, b"false", Json::Bool(false)),
        Some(b'n') => parse_lit(c, b"null", Json::Null),
        Some(b'-') | Some(b'0'..=b'9') => Ok(Json::Number(parse_number(c)?)),
        Some(other) => Err(c.err(format!("unexpected byte 0x{other:02x}"))),
        None => Err(c.err("unexpected end of input")),
    }
}

fn parse_lit(c: &mut Cursor<'_>, lit: &[u8], value: Json) -> Result<Json, ParseError> {
    for expected in lit {
        match c.bump() {
            Some(ch) if ch == *expected => {}
            _ => return Err(c.err(format!("expected {}", String::from_utf8_lossy(lit)))),
        }
    }
    Ok(value)
}

fn parse_object(c: &mut Cursor<'_>) -> Result<Json, ParseError> {
    c.bump(); // `{`
    c.skip_ws();
    let mut pairs = Vec::new();
    if c.peek() == Some(b'}') {
        c.bump();
        return Ok(Json::Object(pairs));
    }
    loop {
        c.skip_ws();
        if c.peek() != Some(b'"') {
            return Err(c.err("expected object key string"));
        }
        let key = parse_string(c)?;
        c.skip_ws();
        if c.bump() != Some(b':') {
            return Err(c.err("expected ':' after object key"));
        }
        let value = parse_value(c)?;
        pairs.push((key, value));
        c.skip_ws();
        match c.bump() {
            Some(b',') => {
                c.skip_ws();
                if c.peek() == Some(b'}') {
                    // trailing comma is rejected — strict JSON.
                    return Err(c.err("trailing comma in object"));
                }
            }
            Some(b'}') => break,
            _ => return Err(c.err("expected ',' or '}' in object")),
        }
    }
    Ok(Json::Object(pairs))
}

fn parse_array(c: &mut Cursor<'_>) -> Result<Json, ParseError> {
    c.bump(); // `[`
    c.skip_ws();
    let mut items = Vec::new();
    if c.peek() == Some(b']') {
        c.bump();
        return Ok(Json::Array(items));
    }
    loop {
        items.push(parse_value(c)?);
        c.skip_ws();
        match c.bump() {
            Some(b',') => {
                c.skip_ws();
                if c.peek() == Some(b']') {
                    return Err(c.err("trailing comma in array"));
                }
            }
            Some(b']') => break,
            _ => return Err(c.err("expected ',' or ']' in array")),
        }
    }
    Ok(Json::Array(items))
}

fn parse_string(c: &mut Cursor<'_>) -> Result<String, ParseError> {
    if c.bump() != Some(b'"') {
        return Err(c.err("expected string"));
    }
    let mut out = String::new();
    loop {
        match c.bump() {
            None => return Err(c.err("unterminated string")),
            Some(b'"') => return Ok(out),
            Some(b'\\') => match c.bump() {
                Some(b'"') => out.push('"'),
                Some(b'\\') => out.push('\\'),
                Some(b'/') => out.push('/'),
                Some(b'b') => out.push('\u{0008}'),
                Some(b'f') => out.push('\u{000c}'),
                Some(b'n') => out.push('\n'),
                Some(b'r') => out.push('\r'),
                Some(b't') => out.push('\t'),
                Some(b'u') => {
                    let mut hex = 0u32;
                    for _ in 0..4 {
                        let h = c.bump().ok_or_else(|| c.err("bad \\u escape"))?;
                        hex = (hex << 4)
                            | hex_val(h).ok_or_else(|| c.err("bad \\u hex digit"))?;
                    }
                    let ch = char::from_u32(hex).ok_or_else(|| c.err("invalid unicode"))?;
                    out.push(ch);
                }
                Some(other) => {
                    return Err(c.err(format!("unknown escape 0x{other:02x}")));
                }
                None => return Err(c.err("unterminated escape")),
            },
            Some(ch) => {
                // Re-decode UTF-8 from the source so multi-byte chars survive.
                if ch < 0x80 {
                    out.push(ch as char);
                } else {
                    // step back one byte and consume a full char from src.
                    c.i -= 1;
                    let rest = &c.src[c.i..];
                    let ch = rest.chars().next().ok_or_else(|| c.err("bad utf-8"))?;
                    out.push(ch);
                    c.i += ch.len_utf8();
                }
            }
        }
    }
}

fn hex_val(b: u8) -> Option<u32> {
    match b {
        b'0'..=b'9' => Some((b - b'0') as u32),
        b'a'..=b'f' => Some((b - b'a' + 10) as u32),
        b'A'..=b'F' => Some((b - b'A' + 10) as u32),
        _ => None,
    }
}

fn parse_number(c: &mut Cursor<'_>) -> Result<String, ParseError> {
    let start = c.i;
    if c.peek() == Some(b'-') {
        c.bump();
    }
    match c.peek() {
        Some(b'0') => {
            c.bump();
        }
        Some(b'1'..=b'9') => {
            while matches!(c.peek(), Some(b'0'..=b'9')) {
                c.bump();
            }
        }
        _ => return Err(c.err("invalid number")),
    }
    if c.peek() == Some(b'.') {
        c.bump();
        if !matches!(c.peek(), Some(b'0'..=b'9')) {
            return Err(c.err("invalid fraction"));
        }
        while matches!(c.peek(), Some(b'0'..=b'9')) {
            c.bump();
        }
    }
    if matches!(c.peek(), Some(b'e') | Some(b'E')) {
        c.bump();
        if matches!(c.peek(), Some(b'+') | Some(b'-')) {
            c.bump();
        }
        if !matches!(c.peek(), Some(b'0'..=b'9')) {
            return Err(c.err("invalid exponent"));
        }
        while matches!(c.peek(), Some(b'0'..=b'9')) {
            c.bump();
        }
    }
    Ok(c.src[start..c.i].to_string())
}

fn emit_yaml(value: &Json, indent: usize, top: bool, out: &mut String) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Number(n) => out.push_str(n),
        Json::String(s) => emit_plain_or_quoted(s, out),
        Json::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 || !top {
                    out.push('\n');
                    push_indent(out, indent);
                }
                out.push_str("- ");
                emit_yaml_nested(item, indent + 2, out);
            }
        }
        Json::Object(pairs) => {
            if pairs.is_empty() {
                out.push_str("{}");
                return;
            }
            for (idx, (key, val)) in pairs.iter().enumerate() {
                if idx > 0 || !top {
                    out.push('\n');
                    push_indent(out, indent);
                }
                emit_plain_or_quoted(key, out);
                match val {
                    Json::Object(p) if !p.is_empty() => {
                        out.push(':');
                        emit_yaml(val, indent + 2, false, out);
                    }
                    Json::Array(a) if !a.is_empty() => {
                        out.push(':');
                        emit_yaml(val, indent + 2, false, out);
                    }
                    _ => {
                        out.push_str(": ");
                        emit_yaml(val, indent + 2, false, out);
                    }
                }
            }
        }
    }
}

fn emit_yaml_nested(value: &Json, indent: usize, out: &mut String) {
    match value {
        Json::Object(pairs) if !pairs.is_empty() => {
            // First key shares the `- ` line; the rest indent under it.
            for (idx, (key, val)) in pairs.iter().enumerate() {
                if idx > 0 {
                    out.push('\n');
                    push_indent(out, indent);
                }
                emit_plain_or_quoted(key, out);
                match val {
                    Json::Object(p) if !p.is_empty() => {
                        out.push(':');
                        emit_yaml(val, indent + 2, false, out);
                    }
                    Json::Array(a) if !a.is_empty() => {
                        out.push(':');
                        emit_yaml(val, indent + 2, false, out);
                    }
                    _ => {
                        out.push_str(": ");
                        emit_yaml(val, indent + 2, false, out);
                    }
                }
            }
        }
        Json::Array(items) if !items.is_empty() => {
            emit_yaml(value, indent, false, out);
        }
        _ => emit_yaml(value, indent, false, out),
    }
}

fn push_indent(out: &mut String, n: usize) {
    for _ in 0..n {
        out.push(' ');
    }
}

fn emit_plain_or_quoted(s: &str, out: &mut String) {
    if is_plain_scalar(s) {
        out.push_str(s);
    } else {
        out.push('"');
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    out.push_str(&format!("\\u{:04x}", c as u32));
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
}

fn is_plain_scalar(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let lower = s.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "true" | "false" | "null" | "yes" | "no" | "on" | "off" | "y" | "n" | "~"
    ) {
        return false;
    }
    if looks_like_number(s) {
        return false;
    }
    let first = s.as_bytes()[0];
    if matches!(
        first,
        b'-' | b'?'
            | b':'
            | b'{'
            | b'}'
            | b'['
            | b']'
            | b','
            | b'&'
            | b'*'
            | b'!'
            | b'|'
            | b'>'
            | b'\''
            | b'"'
            | b'%'
            | b'@'
            | b'`'
            | b'#'
    ) {
        return false;
    }
    !s.chars()
        .any(|c| c.is_whitespace() || c == ':' || c == '#' || c == '&' || c == ',')
}

fn looks_like_number(s: &str) -> bool {
    let mut c = Cursor::new(s);
    if parse_number(&mut c).is_err() {
        return false;
    }
    c.eof()
}

fn emit_json(value: &Json, out: &mut String) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Number(n) => out.push_str(n),
        Json::String(s) => {
            out.push('"');
            for ch in s.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Json::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                emit_json(item, out);
            }
            out.push(']');
        }
        Json::Object(pairs) => {
            out.push('{');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                emit_json(&Json::String(k.clone()), out);
                out.push(':');
                emit_json(v, out);
            }
            out.push('}');
        }
    }
}

/// Minimal block-YAML reader covering exactly the dialect `emit_yaml` writes.
fn parse_yaml(src: &str) -> Result<Json, String> {
    let lines: Vec<&str> = src.lines().collect();
    if lines.iter().all(|l| l.trim().is_empty()) {
        return Err("empty yaml".into());
    }
    let mut idx = 0usize;
    // Skip leading blanks.
    while idx < lines.len() && lines[idx].trim().is_empty() {
        idx += 1;
    }
    parse_yaml_node(&lines, &mut idx, 0)
}

fn indent_of(line: &str) -> usize {
    line.chars().take_while(|c| *c == ' ').count()
}

fn parse_yaml_node(lines: &[&str], idx: &mut usize, min_indent: usize) -> Result<Json, String> {
    if *idx >= lines.len() {
        return Ok(Json::Null);
    }
    let line = lines[*idx];
    if line.trim().is_empty() {
        *idx += 1;
        return parse_yaml_node(lines, idx, min_indent);
    }
    let indent = indent_of(line);
    if indent < min_indent {
        return Ok(Json::Null);
    }
    let trimmed = line[indent..].trim_end();
    if trimmed.starts_with("- ") || trimmed == "-" {
        return parse_yaml_array(lines, idx, indent);
    }
    if trimmed.contains(':') {
        return parse_yaml_object(lines, idx, indent);
    }
    *idx += 1;
    parse_yaml_scalar(trimmed)
}

fn parse_yaml_object(
    lines: &[&str],
    idx: &mut usize,
    indent: usize,
) -> Result<Json, String> {
    let mut pairs = Vec::new();
    while *idx < lines.len() {
        let line = lines[*idx];
        if line.trim().is_empty() {
            *idx += 1;
            continue;
        }
        let i = indent_of(line);
        if i < indent {
            break;
        }
        if i > indent {
            return Err(format!("unexpected indent at line {}", *idx + 1));
        }
        let trimmed = line[i..].trim_end();
        if trimmed.starts_with("- ") {
            break;
        }
        let colon = trimmed
            .find(':')
            .ok_or_else(|| format!("expected key: at line {}", *idx + 1))?;
        let key_raw = trimmed[..colon].trim();
        let key = unquote(key_raw);
        let rest = trimmed[colon + 1..].trim();
        *idx += 1;
        let value = if rest.is_empty() {
            if *idx < lines.len() {
                let next_indent = lines[*idx]
                    .find(|c: char| c != ' ')
                    .map(|n| {
                        if lines[*idx].trim().is_empty() {
                            indent + 2
                        } else {
                            n
                        }
                    })
                    .unwrap_or(indent + 2);
                if next_indent > indent {
                    parse_yaml_node(lines, idx, indent + 1)?
                } else {
                    Json::Null
                }
            } else {
                Json::Null
            }
        } else {
            parse_yaml_scalar(rest)?
        };
        pairs.push((key, value));
    }
    Ok(Json::Object(pairs))
}

fn parse_yaml_array(lines: &[&str], idx: &mut usize, indent: usize) -> Result<Json, String> {
    let mut items = Vec::new();
    while *idx < lines.len() {
        let line = lines[*idx];
        if line.trim().is_empty() {
            *idx += 1;
            continue;
        }
        let i = indent_of(line);
        if i < indent {
            break;
        }
        let trimmed = line[i..].trim_end();
        if !trimmed.starts_with('-') {
            break;
        }
        let rest = trimmed.trim_start_matches('-').trim();
        *idx += 1;
        if rest.is_empty() {
            items.push(parse_yaml_node(lines, idx, indent + 1)?);
        } else if rest.contains(':') {
            // Inline first key of an object item.
            let fake = format!("{:indent$}{rest}", "", indent = indent + 2);
            // Re-parse this synthetic line plus any following indented keys.
            let mut buf: Vec<String> = vec![fake];
            while *idx < lines.len() {
                let nxt = lines[*idx];
                if nxt.trim().is_empty() {
                    break;
                }
                let ni = indent_of(nxt);
                if ni >= indent + 2 && !nxt.trim_start().starts_with("- ") {
                    buf.push(nxt.to_string());
                    *idx += 1;
                } else {
                    break;
                }
            }
            let refs: Vec<&str> = buf.iter().map(|s| s.as_str()).collect();
            let mut j = 0;
            items.push(parse_yaml_object(&refs, &mut j, indent + 2)?);
        } else {
            items.push(parse_yaml_scalar(rest)?);
        }
    }
    Ok(Json::Array(items))
}

fn parse_yaml_scalar(raw: &str) -> Result<Json, String> {
    if raw == "[]" {
        return Ok(Json::Array(Vec::new()));
    }
    if raw == "{}" {
        return Ok(Json::Object(Vec::new()));
    }
    if raw == "null" || raw == "~" {
        return Ok(Json::Null);
    }
    if raw == "true" {
        return Ok(Json::Bool(true));
    }
    if raw == "false" {
        return Ok(Json::Bool(false));
    }
    if looks_like_number(raw) {
        return Ok(Json::Number(raw.to_string()));
    }
    Ok(Json::String(unquote(raw)))
}

fn unquote(s: &str) -> String {
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        let inner = &s[1..s.len() - 1];
        let mut out = String::new();
        let mut chars = inner.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\\' {
                match chars.next() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some(other) => out.push(other),
                    None => {}
                }
            } else {
                out.push(ch);
            }
        }
        out
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattens_nested_object() {
        let json = r#"{"user":{"name":"alice","age":30},"ok":true}"#;
        let yaml = DataConverter::new().convert(json);
        assert!(yaml.contains("user:"));
        assert!(yaml.contains("name: alice"));
        assert!(yaml.contains("age: 30"));
        assert!(!yaml.contains('{'));
        assert!(!yaml.contains('"'));
    }

    #[test]
    fn quotes_ambiguous_strings() {
        let json = r#"{"note":"15.4","flag":"true"}"#;
        let yaml = DataConverter::new().convert(json);
        assert!(yaml.contains("\"15.4\""));
        assert!(yaml.contains("\"true\""));
    }

    #[test]
    fn round_trip_nested() {
        let json = r#"{"a":{"b":[1,2,{"c":"x"}]},"d":null}"#;
        let conv = DataConverter::new();
        let yaml = conv.convert(json);
        let back = conv.yaml_to_json(&yaml).expect("yaml");
        let original = parse_json(json).unwrap();
        let recovered = parse_json(&back).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn leaves_non_json_alone() {
        let src = "fn main() {}\n";
        assert_eq!(DataConverter::new().convert(src), src);
    }
}
