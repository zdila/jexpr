use rustc_hash::FxHashMap;
use serde_json::Number;
use std::{collections::HashMap, f64, hash::BuildHasher, sync::Arc};

pub use serde_json::Value;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("filter expression is not a valid JSON: {0}")]
    ParseError(String),

    #[error("Evaluation error: {0}")]
    EvaluationError(String),

    #[error("Invalid number")]
    InvalidNumber,

    #[error("Function evaluation error: {0}")]
    FunctionEvaluationError(Box<dyn std::error::Error + Send + Sync>),
}

impl From<Box<dyn std::error::Error + Send + Sync>> for Error {
    fn from(value: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self::FunctionEvaluationError(value)
    }
}

pub type Function<'a, C = FxHashMap<String, String>> =
    Box<dyn Fn(&C, Value) -> Result<Value, Error> + Send + Sync + 'a>;

pub trait Tags {
    fn get(&self, key: &str) -> Option<String>;
    fn contains_key(&self, key: &str) -> bool;
}

impl<S: BuildHasher> Tags for HashMap<String, String, S> {
    fn get(&self, key: &str) -> Option<String> {
        self.get(key).cloned()
    }

    fn contains_key(&self, key: &str) -> bool {
        self.contains_key(key)
    }
}

pub type Expression<'a, C> = Box<dyn Fn(&C) -> Result<Value, Error> + Send + Sync + 'a>;

pub type ParseResult<'a, C> = Result<Expression<'a, C>, Error>;

/// # Errors
///
/// Parse `MapLibre` Expression (incomplete implementation)
/// and return function that evaluates the expression with the provided context
pub fn parse<'a, C: Tags + 'a>(expression: &str) -> ParseResult<'a, C> {
    compile(
        Arc::new(HashMap::new()),
        serde_json::from_str(expression).map_err(|e| Error::ParseError(e.to_string()))?,
    )
}

/// # Errors
///
/// Parse `MapLibre` Expression (incomplete implementation)
/// and return function that evaluates the expression with the provided context
pub fn parse_with_functions<'a, C: Tags + 'a, S: BuildHasher + Sync + Send + Default + 'a>(
    expression: &str,
    functions: HashMap<String, Function<'a, C>, S>,
) -> ParseResult<'a, C> {
    compile(
        Arc::new(functions),
        serde_json::from_str(expression).map_err(|e| Error::ParseError(e.to_string()))?,
    )
}

#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::too_many_lines)] // TODO
fn compile<'a, C: Tags + 'a, S: BuildHasher + Sync + Send + 'a>(
    functions: Arc<HashMap<String, Function<'a, C>, S>>,
    value: Value,
) -> ParseResult<'a, C> {
    let Value::Array(array) = value else {
        // literals

        return Ok(Box::new(move |_ctx| Ok(value.clone())));
    };

    let mut iter = array.into_iter();

    let operator = iter.next();

    let Some(Value::String(str)) = operator else {
        return Err(Error::EvaluationError(
            "operator must be a string literal".into(),
        ));
    };

    let mut get_next_arg = || {
        let next = iter
            .next()
            .ok_or_else(|| Error::EvaluationError("missing argument".into()))?;

        compile(Arc::clone(&functions), next)
    };

    match str.as_str() {
        "!" => {
            let arg = get_next_arg()?;

            Ok(Box::new(move |ctx| match arg(ctx)? {
                Value::Bool(bool) => Ok(Value::Bool(!bool)),
                _ => Err(Error::EvaluationError("not a bool".into())),
            }))
        }

        "to-string" => {
            let arg = get_next_arg()?;

            Ok(Box::new(move |ctx| {
                Ok(Value::String(match arg(ctx)? {
                    Value::Bool(bool) => (if bool { "true" } else { "false" }).to_owned(),
                    Value::Null => String::new(),
                    Value::String(s) => s,
                    Value::Number(n) => n.to_string(),
                    _ => Err(Error::EvaluationError("can't convert to string".into()))?,
                }))
            }))
        }

        "to-boolean" => {
            let arg = get_next_arg()?;

            Ok(Box::new(move |ctx| {
                Ok(Value::Bool(match arg(ctx)? {
                    Value::Bool(bool) => bool,
                    Value::String(s) => !s.is_empty(),
                    Value::Number(number) => {
                        let number = number
                            .as_f64()
                            .ok_or(Error::EvaluationError("error converting to f64".into()))?;
                        !number.is_nan() && number != 0.
                    }
                    _ => true,
                }))
            }))
        }

        "to-number" => {
            let arg = get_next_arg()?;

            Ok(Box::new(move |ctx| {
                Ok(Value::Number(match arg(ctx)? {
                    Value::Null => Number::from_f64(0.).unwrap(),
                    Value::Number(n) => n,
                    Value::Bool(bool) => Number::from_f64(if bool { 1. } else { 0. }).unwrap(),
                    Value::String(s) => {
                        Number::from_f64(s.parse::<f64>().map_err(|_| Error::InvalidNumber)?)
                            .ok_or_else(|| Error::EvaluationError("unsupported number".into()))?
                    }
                    _ => Err(Error::EvaluationError("can't convert to number".into()))?,
                }))
            }))
        }

        "length" => {
            let arg = get_next_arg()?;

            Ok(Box::new(move |ctx| {
                Ok(Value::Number(match arg(ctx)? {
                    Value::Array(a) => Number::from(a.len()),
                    Value::String(s) => Number::from(s.len()),
                    _ => Err(Error::EvaluationError(
                        "`length` works only on arrays or strings".into(),
                    ))?,
                }))
            }))
        }

        "==" => {
            let (arg_1, arg_2) = (get_next_arg()?, get_next_arg()?);

            Ok(Box::new(move |ctx| {
                Ok(Value::Bool(arg_1(ctx)? == arg_2(ctx)?))
            }))
        }

        "!=" => {
            let (arg_1, arg_2) = (get_next_arg()?, get_next_arg()?);

            Ok(Box::new(move |ctx| {
                Ok(Value::Bool(arg_1(ctx)? != arg_2(ctx)?))
            }))
        }

        ">" | "<" | ">=" | "<=" => {
            let (arg_1, arg_2) = (get_next_arg()?, get_next_arg()?);

            Ok(Box::new(move |ctx| {
                let v1 = arg_1(ctx)?;
                let v2 = arg_2(ctx)?;

                Ok(Value::Bool(match (v1, v2) {
                    (Value::String(s1), Value::String(s2)) => match str.as_str() {
                        ">" => s1 > s2,
                        "<" => s1 < s2,
                        ">=" => s1 >= s2,
                        "<=" => s1 <= s2,
                        _ => unreachable!(),
                    },
                    (Value::Number(n1), Value::Number(n2)) => {
                        let n1 = n1
                            .as_f64()
                            .ok_or(Error::EvaluationError("error converting from f64".into()))?;
                        let n2 = n2
                            .as_f64()
                            .ok_or(Error::EvaluationError("error converting from f64".into()))?;

                        match str.as_str() {
                            ">" => n1 > n2,
                            "<" => n1 < n2,
                            ">=" => n1 >= n2,
                            "<=" => n1 <= n2,
                            _ => unreachable!(),
                        }
                    }
                    _ => Err(Error::EvaluationError("exprected number or string".into()))?,
                }))
            }))
        }

        "abs" | "acos" | "asin" | "atan" | "ceil" | "cos" | "floor" | "ln" | "log10" | "log2"
        | "round" | "sin" | "sqrt" | "tan" => {
            let arg = get_next_arg()?;

            Ok(Box::new(move |ctx| {
                let value = arg(ctx)?;

                match value {
                    Value::Number(number) => {
                        let number = number
                            .as_f64()
                            .ok_or(Error::EvaluationError("error converting from f64".into()))?;

                        let value = match str.as_str() {
                            "abs" => number.abs(),
                            "acos" => number.acos(),
                            "asin" => number.asin(),
                            "atan" => number.atan(),
                            "ceil" => number.ceil(),
                            "cos" => number.cos(),
                            "floor" => number.floor(),
                            "ln" => number.ln(),
                            "log10" => number.log10(),
                            "log2" => number.log2(),
                            "round" => number.round(),
                            "sin" => number.sin(),
                            "sqrt" => number.sqrt(),
                            "tan" => number.tan(),
                            _ => unreachable!(),
                        };

                        Ok(Value::Number(Number::from_f64(value).ok_or(
                            Error::EvaluationError("error converting to f64".into()),
                        )?))
                    }
                    _ => Err(Error::EvaluationError("expected number".into()))?,
                }
            }))
        }

        "e" | "pi" | "ln2" | "ln10" => Ok(Box::new(move |_| {
            let value = match str.as_str() {
                "e" => f64::consts::E,
                "pi" => f64::consts::PI,
                "ln2" => f64::consts::LN_2,
                "ln10" => f64::consts::LN_10,
                _ => unreachable!(),
            };

            Ok(Value::Number(Number::from_f64(value).ok_or(
                Error::EvaluationError("error converting to f64".into()),
            )?))
        })),

        "+" | "*" | "min" | "max" => {
            let mut args = Vec::new();

            for arg in iter {
                args.push(compile(Arc::clone(&functions), arg)?);
            }

            Ok(Box::new(move |ctx| {
                let values = args
                    .iter()
                    .map(|arg| {
                        arg(ctx).and_then(|value| match value {
                            Value::Number(num) => num
                                .as_f64()
                                .ok_or(Error::EvaluationError("error converting to f64".into())),
                            _ => Err(Error::EvaluationError("expected number".into())),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let result = match str.as_str() {
                    "+" => values.into_iter().fold(0.0, |acc, v| acc + v),
                    "*" => values.into_iter().fold(1.0, |acc, v| acc * v),
                    "min" => values.into_iter().fold(f64::INFINITY, f64::min),
                    "max" => values.into_iter().fold(f64::NEG_INFINITY, f64::max),
                    _ => unreachable!(),
                };

                Ok(Value::Number(Number::from_f64(result).ok_or(
                    Error::EvaluationError("error converting to f64".into()),
                )?))
            }))
        }

        "-" | "/" | "%" | "random" => {
            let (arg_1, arg_2) = (get_next_arg()?, get_next_arg()?);

            Ok(Box::new(move |ctx| {
                let v1 = arg_1(ctx)?;
                let v2 = arg_2(ctx)?;

                Ok(match (v1, v2) {
                    (Value::Number(n1), Value::Number(n2)) => {
                        let n1 = n1
                            .as_f64()
                            .ok_or(Error::EvaluationError("error converting from f64".into()))?;

                        let n2 = n2
                            .as_f64()
                            .ok_or(Error::EvaluationError("error converting from f64".into()))?;

                        // match str.as_str()
                        Value::Number(
                            Number::from_f64(match str.as_str() {
                                "-" => n1 - n2,
                                "/" => n1 / n2,
                                "%" => n1 % n2,
                                "random" => rand::random_range(n1..n2), // TODO seed is not supported yet
                                _ => unreachable!(),
                            })
                            .ok_or(Error::EvaluationError("error converting to f64".into()))?,
                        )
                    }
                    _ => return Err(Error::EvaluationError("expected number".into())),
                })
            }))
        }

        "upcase" | "downcase" => {
            let arg = get_next_arg()?;

            Ok(Box::new(move |ctx| {
                let value = arg(ctx)?;

                match value {
                    Value::String(string) => Ok(Value::String(match str.as_str() {
                        "upcase" => string.to_uppercase(),
                        "downcase" => string.to_lowercase(),
                        _ => unreachable!(),
                    })),
                    _ => Err(Error::EvaluationError("expected string".into())),
                }
            }))
        }

        "get" => {
            let arg1 = get_next_arg()?;

            let arg2 = iter
                .next()
                .map(|arg2| compile(Arc::clone(&functions), arg2))
                .transpose()?;

            Ok(Box::new(move |ctx| match arg1(ctx)? {
                Value::String(key) => {
                    if let Some(arg2) = &arg2 {
                        match arg2(ctx)? {
                            Value::Object(arg2) => {
                                Ok(arg2.get(&key).map_or(Value::Null, ToOwned::to_owned))
                            }
                            _ => Err(Error::EvaluationError(
                                "`has` second argument must be an object".into(),
                            )),
                        }
                    } else {
                        Ok(ctx.get(&key).map_or(Value::Null, Value::String))
                    }
                }
                _ => Err(Error::EvaluationError(
                    "`get` argument must be a string".into(),
                ))?,
            }))
        }

        "has" => {
            let arg1 = get_next_arg()?;

            let arg2 = iter
                .next()
                .map(|arg2| compile(Arc::clone(&functions), arg2))
                .transpose()?;

            Ok(Box::new(move |ctx| match arg1(ctx)? {
                Value::String(key) => {
                    if let Some(arg2) = &arg2 {
                        match arg2(ctx)? {
                            Value::Object(arg2) => Ok(Value::Bool(arg2.contains_key(&key))),
                            _ => Err(Error::EvaluationError(
                                "`has` second argument must be an object".into(),
                            )),
                        }
                    } else {
                        Ok(Value::Bool(ctx.contains_key(&key)))
                    }
                }
                _ => Err(Error::EvaluationError(
                    "`has` first argument must be a string".into(),
                ))?,
            }))
        }

        "at" => {
            let arg1 = get_next_arg()?;

            let arg2 = get_next_arg()?;

            Ok(Box::new(move |ctx| match arg1(ctx)? {
                Value::Number(index) => {
                    let index = index
                        .as_u64()
                        .ok_or_else(|| Error::EvaluationError("can't convert to u64".into()))?;

                    let index = usize::try_from(index).unwrap();

                    match arg2(ctx)? {
                        Value::Array(arg2) => {
                            Ok(arg2.get(index).map_or(Value::Null, ToOwned::to_owned))
                        }
                        _ => Err(Error::EvaluationError(
                            "`has` second argument must be an array".into(),
                        )),
                    }
                }
                _ => Err(Error::EvaluationError(
                    "`at` first argument must be a number".into(),
                ))?,
            }))
        }

        "typeof" => {
            let arg = get_next_arg()?;

            Ok(Box::new(move |ctx| {
                Ok(Value::String(
                    (match arg(ctx)? {
                        Value::Null => "null",
                        Value::Bool(_) => "boolean",
                        Value::Number(_) => "number",
                        Value::String(_) => "string",
                        Value::Array(_) => "array",
                        Value::Object(_) => "object",
                    })
                    .to_string(),
                ))
            }))
        }

        "any" => {
            let mut args = Vec::new();

            for arg in iter {
                args.push(compile(Arc::clone(&functions), arg)?);
            }

            Ok(Box::new(move |ctx| {
                for arg in &args {
                    let value = arg(ctx)?;

                    match value {
                        Value::Bool(bool) => {
                            if bool {
                                return Ok(value);
                            }
                        }
                        _ => return Err(Error::EvaluationError("not a bool".into())),
                    }
                }

                Ok(Value::Bool(false))
            }))
        }

        "all" => {
            let mut args = Vec::new();

            for arg in iter {
                args.push(compile(Arc::clone(&functions), arg)?);
            }

            Ok(Box::new(move |ctx| {
                for arg in &args {
                    match arg(ctx)? {
                        Value::Bool(bool) => {
                            if !bool {
                                return Ok(Value::Bool(false));
                            }
                        }
                        _ => return Err(Error::EvaluationError("not a bool".into())),
                    }
                }

                Ok(Value::Bool(true))
            }))
        }

        "coalesce" => {
            let mut args = Vec::new();

            for arg in iter {
                args.push(compile(Arc::clone(&functions), arg)?);
            }

            Ok(Box::new(move |ctx| {
                for arg in &args {
                    let value = arg(ctx)?;

                    if !value.is_null() {
                        return Ok(value);
                    }
                }

                Ok(Value::Null)
            }))
        }

        "case" => {
            let mut args = Vec::new();

            for arg in iter {
                args.push(compile(Arc::clone(&functions), arg)?);
            }

            Ok(Box::new(move |ctx| {
                let mut iter = args.iter();

                loop {
                    let cond = match iter.next() {
                        None => {
                            return Err(Error::EvaluationError("missing default value".into()));
                        }
                        Some(arg) => arg(ctx)?,
                    };

                    let value = match iter.next() {
                        None => return Ok(cond),
                        Some(arg) => arg(ctx)?,
                    };

                    if let Value::Bool(bool) = cond {
                        if bool {
                            return Ok(value);
                        }
                    } else {
                        return Err(Error::EvaluationError(
                            "case condition is not a bool".into(),
                        ));
                    }
                }
            }))
        }

        "concat" => {
            let mut args = Vec::new();

            for arg in iter {
                args.push(compile(Arc::clone(&functions), arg)?);
            }

            Ok(Box::new(move |ctx| {
                args.iter()
                    .map(|arg| {
                        arg(ctx).and_then(|value| match value {
                            Value::String(string) => Ok(string),
                            _ => Err(Error::EvaluationError("expected string".into())),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(|strings| Value::String(strings.concat()))
            }))
        }

        // TODO add more operators
        fn_name => {
            if !functions.contains_key(fn_name) {
                return Err(Error::EvaluationError(format!(
                    "no such operator or function: {str}"
                )));
            };

            let mut args = Vec::new();

            for arg in iter {
                args.push(compile(Arc::clone(&functions), arg)?);
            }

            let functions = Arc::clone(&functions);

            let fn_name = fn_name.to_string();

            Ok(Box::new(move |ctx| {
                let function = functions.get(&fn_name).unwrap();

                let mut values = Vec::new();

                for arg in &args {
                    let value = arg(ctx)?;

                    values.push(value);
                }

                function(ctx, Value::Array(values))
                    .map_err(|e| Error::FunctionEvaluationError(e.into()))
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use super::*;

    #[test]
    fn test() -> Result<(), Error> {
        assert_eq!(parse("2")?(&FxHashMap::default())?, 2);

        assert_eq!(parse(r#"["+", 2, 2]"#)?(&FxHashMap::default())?, 4.);

        assert_eq!(parse(r#"["+", "2", "2"]"#)?(&FxHashMap::default())?, "22");

        assert_eq!(parse(r#"[">", 3, -2]"#)?(&FxHashMap::default())?, true);

        assert_eq!(
            parse(r#"["all", true, false]"#)?(&FxHashMap::default())?,
            false
        );

        assert_eq!(
            parse(r#"["all", true, true]"#)?(&FxHashMap::default())?,
            true
        );

        assert_eq!(
            parse(r#"["any", true, false]"#)?(&FxHashMap::default())?,
            true
        );

        assert_eq!(
            parse(r#"["any", false, false]"#)?(&FxHashMap::default())?,
            false
        );

        assert_eq!(
            parse(r#"["coalesce", null, "foo", 42]"#)?(&FxHashMap::default())?,
            "foo"
        );

        assert_eq!(
            parse(r#"["case", false, "foo", true, "bar", "baz"]"#)?(&FxHashMap::default())?,
            "bar"
        );

        let mut ctx = FxHashMap::default();

        ctx.insert(String::from("name"), String::from("martin"));

        assert_eq!(parse(r#"["==", ["get", "name"], "martin"]"#)?(&ctx)?, true);

        Ok(())
    }
}
