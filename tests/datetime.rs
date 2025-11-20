use otters::datetime::{parse_datetime_millis, parse_datetime_millis_with_format};

#[test]
fn parses_default_formats() {
    assert!(parse_datetime_millis("2024-01-02T03:04:05Z").is_ok());
    assert!(parse_datetime_millis("2024-01-02").is_ok());
    assert!(parse_datetime_millis("2024-01-02 03:04:05").is_ok());
}

#[test]
fn parses_custom_format() {
    let millis = parse_datetime_millis_with_format("02/01/2024", "%d/%m/%Y").unwrap();
    assert_eq!(millis, 1704153600000);
}

#[test]
fn errors_include_input() {
    let err = parse_datetime_millis("nope").unwrap_err();
    assert!(err.to_string().contains("nope"));
}
