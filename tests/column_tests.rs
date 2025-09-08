use otters::col::{Column, ColumnError};
use otters::type_utils::DataType;

#[cfg(test)]
mod column_tests {
    use super::*;

    #[test]
    fn test_column_creation() {
        let col = Column::new("test", DataType::Int32);
        assert_eq!(col.name(), "test");
        assert_eq!(col.dtype(), DataType::Int32);
        assert_eq!(col.len(), 0);
        assert!(col.is_empty());
    }

    #[test]
    fn test_unified_push_int32() {
        let mut col = Column::new("integers", DataType::Int32);

        // Test direct value
        assert!(col.push(42).is_ok());
        assert_eq!(col.len(), 1);

        // Test optional value
        assert!(col.push(Some(100)).is_ok());
        assert_eq!(col.len(), 2);

        // Test None value
        assert!(col.push(None::<i32>).is_ok());
        assert_eq!(col.len(), 3);

        // Verify null mask
        let null_mask = col.null_mask();
        assert!(!null_mask[0]); // false = not null
        assert!(!null_mask[1]); // false = not null
        assert!(null_mask[2]); // true = null
    }

    #[test]
    fn test_unified_push_int64() {
        let mut col = Column::new("big_integers", DataType::Int64);

        assert!(col.push(42i64).is_ok());
        assert!(col.push(Some(100i64)).is_ok());
        assert!(col.push(None::<i64>).is_ok());
        assert_eq!(col.len(), 3);
    }

    #[test]
    fn test_unified_push_float32() {
        let mut col = Column::new("floats", DataType::Float32);

        assert!(col.push(std::f32::consts::PI).is_ok());
        assert!(col.push(Some(2.71f32)).is_ok());
        assert!(col.push(None::<f32>).is_ok());
        assert_eq!(col.len(), 3);
    }

    #[test]
    fn test_unified_push_float64() {
        let mut col = Column::new("doubles", DataType::Float64);

        assert!(col.push(std::f64::consts::PI).is_ok());
        assert!(col.push(Some(std::f64::consts::E)).is_ok());
        assert!(col.push(None::<f64>).is_ok());
        assert_eq!(col.len(), 3);
    }

    #[test]
    fn test_unified_push_string() {
        let mut col = Column::new("strings", DataType::String);

        // Test &str
        assert!(col.push("hello").is_ok());

        // Test String
        assert!(col.push("world".to_string()).is_ok());

        // Test Option<String>
        assert!(col.push(Some("rust".to_string())).is_ok());

        // Test Option<&str>
        assert!(col.push(Some("programming")).is_ok());

        // Test None
        assert!(col.push(None::<&str>).is_ok());

        assert_eq!(col.len(), 5);
    }

    #[test]
    fn test_unified_push_datetime_auto_format() {
        let mut col = Column::new("timestamps", DataType::DateTime);

        // Test ISO format
        assert!(col.push("2024-01-15T10:30:00Z").is_ok());

        // Test simple format
        assert!(col.push("2024-02-20 15:45:30").is_ok());

        // Test date only
        assert!(col.push("2024-03-10").is_ok());

        // Test None
        assert!(col.push(None::<&str>).is_ok());

        assert_eq!(col.len(), 4);
    }

    #[test]
    fn test_unified_push_datetime_custom_format() {
        let mut col = Column::new("events", DataType::DateTime).with_datetime_fmt("%m/%d/%Y");

        assert!(col.push("01/15/2024").is_ok());
        assert!(col.push("02/20/2024").is_ok());
        assert!(col.push(None::<&str>).is_ok());

        assert_eq!(col.len(), 3);
    }

    #[test]
    fn test_type_mismatch_errors() {
        let mut int_col = Column::new("integers", DataType::Int32);

        // This should work - i32 to Int32 column
        assert!(int_col.push(42).is_ok());

        // Create a float column and try to push an int value through the old API
        let mut float_col = Column::new("floats", DataType::Float32);

        // This should work - f32 to Float32 column
        assert!(float_col.push(std::f32::consts::PI).is_ok());

        // Test that the unified API prevents type mismatches at compile time
        // (These wouldn't compile, which is what we want)
        // int_col.push(3.14f32); // Compile error: no impl From<f32> for ColumnValue compatible with Int32
    }

    #[test]
    fn test_from_method_int32() {
        let values = vec![1, 2, 3, 4, 5];
        let col = Column::new("integers", DataType::Int32).from(values);

        assert!(col.is_ok());
        let col = col.unwrap();
        assert_eq!(col.len(), 5);
    }

    #[test]
    fn test_from_method_mixed_optionals() {
        let values = vec![Some(1), None, Some(3), None, Some(5)];
        let col = Column::new("mixed", DataType::Int32).from(values);

        assert!(col.is_ok());
        let col = col.unwrap();
        assert_eq!(col.len(), 5);

        let null_mask = col.null_mask();
        assert!(!null_mask[0]); // Some(1) - not null
        assert!(null_mask[1]); // None - null
        assert!(!null_mask[2]); // Some(3) - not null
        assert!(null_mask[3]); // None - null
        assert!(!null_mask[4]); // Some(5) - not null
    }

    #[test]
    fn test_from_method_strings() {
        let values = vec!["Alice", "Bob", "Charlie"];
        let col = Column::new("names", DataType::String).from(values);

        assert!(col.is_ok());
        let col = col.unwrap();
        assert_eq!(col.len(), 3);
    }

    #[test]
    fn test_from_method_datetime_with_format() {
        let values = vec![
            Some("2024-01-15"),
            Some("2024-02-20"),
            None,
            Some("2024-03-10"),
        ];

        let col = Column::new("dates", DataType::DateTime)
            .with_datetime_fmt("%Y-%m-%d")
            .from(values);

        assert!(col.is_ok());
        let col = col.unwrap();
        assert_eq!(col.len(), 4);
    }

    #[test]
    fn test_datetime_parse_errors() {
        let mut col = Column::new("bad_dates", DataType::DateTime);

        // Invalid date format should return error
        let result = col.push("invalid-date-format");
        assert!(result.is_err());

        match result {
            Err(ColumnError::ParseError(_)) => {} // Expected
            _ => panic!("Expected ParseError"),
        }
    }

    #[test]
    fn test_datetime_custom_format_errors() {
        let mut col = Column::new("custom_dates", DataType::DateTime).with_datetime_fmt("%Y-%m-%d");

        // Wrong format should return error
        let result = col.push("01/15/2024"); // MM/dd/yyyy format to YYYY-MM-DD column
        assert!(result.is_err());

        match result {
            Err(ColumnError::ParseError(_)) => {} // Expected
            _ => panic!("Expected ParseError"),
        }
    }

    #[test]
    fn test_mixed_operations() {
        let mut col = Column::new("mixed_ops", DataType::Float64);

        // Mix push and from operations
        assert!(col.push(1.1).is_ok());
        assert!(col.push(Some(2.2)).is_ok());

        let more_values = vec![3.3, 4.4, 5.5];
        let col = col.from(more_values);
        assert!(col.is_ok());
        let mut col = col.unwrap();

        assert!(col.push(None::<f64>).is_ok());

        assert_eq!(col.len(), 6);
    }

    #[test]
    fn test_column_data_access() {
        let col = Column::new("test_data", DataType::Int32)
            .from(vec![1, 2, 3])
            .unwrap();

        // Test data access methods
        let values = col.i32_values();
        assert!(values.is_some());
        let values = values.unwrap();
        // Note: actual values might be sentinel values for nulls,
        // so we just check the length
        assert_eq!(values.len(), 3);

        // Test wrong type access
        assert!(col.f32_values().is_none());
        assert!(col.string_values().is_none());
    }

    #[test]
    fn test_empty_from_operations() {
        let empty_vec: Vec<i32> = vec![];
        let col = Column::new("empty_test", DataType::Int32).from(empty_vec);

        assert!(col.is_ok());
        let col = col.unwrap();
        assert_eq!(col.len(), 0);
        assert!(col.is_empty());
    }

    #[test]
    fn test_large_dataset() {
        let large_values: Vec<i32> = (0..1000).collect();
        let col = Column::new("large", DataType::Int32).from(large_values);

        assert!(col.is_ok());
        let col = col.unwrap();
        assert_eq!(col.len(), 1000);

        // Add more data using new API
        let more_values: Vec<Option<i32>> = (1000..1500).map(Some).collect();
        let col = col.from(more_values);
        assert!(col.is_ok());
        let col = col.unwrap();
        assert_eq!(col.len(), 1500);
    }

    #[test]
    fn test_datetime_from_strings() {
        let mut col = Column::new("dates", DataType::DateTime);

        // Test pushing datetime strings (main use case)
        assert!(col.push("2024-01-15T10:30:00Z").is_ok());
        assert!(col.push("2024-02-20").is_ok());
        assert!(col.push(None::<&str>).is_ok());

        assert_eq!(col.len(), 3);
    }

    #[test]
    fn test_method_chaining() {
        // Test complete method chaining
        let result = Column::new("chained", DataType::Int32)
            .from(vec![1, 2, 3])
            .and_then(|col| col.from(vec![4, 5]));

        assert!(result.is_ok());
        let col = result.unwrap();
        assert_eq!(col.len(), 5);
    }

    #[test]
    fn test_values_method() {
        // Test the unified values() method
        let col = Column::new("test_values", DataType::Int32)
            .from(vec![1, 2, 3, 4, 5])
            .unwrap();

        let values = col.values();
        assert_eq!(values.len(), 5);
        assert!(!values.is_empty());
        assert_eq!(values.data_type(), DataType::Int32);

        // Test with different types
        let float_col = Column::new("float_values", DataType::Float64)
            .from(vec![1.1, 2.2, 3.3])
            .unwrap();

        let float_values = float_col.values();
        assert_eq!(float_values.len(), 3);
        assert_eq!(float_values.data_type(), DataType::Float64);

        // Test with string column
        let str_col = Column::new("string_values", DataType::String)
            .from(vec!["hello", "world"])
            .unwrap();

        let str_values = str_col.values();
        assert_eq!(str_values.len(), 2);
        assert_eq!(str_values.data_type(), DataType::String);
    }
}
