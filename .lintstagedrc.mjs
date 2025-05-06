import { env } from "node:process";

export default {
  "pre-commit": {
    "*.{rs, _1_ }": ["rustfmt --edition 2021 --check"],
    "*.{rs, _2_ }": [
      "bash -c 'cargo clippy -- -D clippy::correctness -D clippy::complexity -D clippy::pedantic -D clippy::nursery -D clippy::perf'",
    ],
  },
}[env.LINT_STAGE] ?? { _nonempty_: [] };
