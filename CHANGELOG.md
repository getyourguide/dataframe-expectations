# Changelog

## [0.3.0](https://github.com/getyourguide/dataframe-expectations/compare/v0.2.0...v0.3.0) (2025-11-09)


### ⚠ BREAKING CHANGES

* The DataFrameExpectationsSuite API has changed. Users must now call .build() before .run().

### Features

* implement builder pattern for expectation suite runner ([66cf5a4](https://github.com/getyourguide/dataframe-expectations/commit/66cf5a4f77bb42cf784946df4250f1d8420c6b4d))


### Bug Fixes

* update release please config to generate simple tags ([185a308](https://github.com/getyourguide/dataframe-expectations/commit/185a308bb8e3582c0d5988e96dd4994beff0a5da))
* update release please config to generate simple tags ([fe767c7](https://github.com/getyourguide/dataframe-expectations/commit/fe767c758d1b02042e8dc449ca924b349e9d5916))


### Documentation

* add Spark session initialization to PySpark examples and update author info ([2b0cf25](https://github.com/getyourguide/dataframe-expectations/commit/2b0cf25363ddf245b9f1c42e01c11fc1e8a5909e))

## [0.2.0](https://github.com/getyourguide/dataframe-expectations/compare/dataframe-expectations-v0.1.1...dataframe-expectations-v0.2.0) (2025-11-08)


### Features

* call expectation functions dynamically, includes registry refactoring ([ecc2328](https://github.com/getyourguide/dataframe-expectations/commit/ecc23287ab47969711383176dae40252fcd27460))


### Bug Fixes

* added more badges to readme ([5447db1](https://github.com/getyourguide/dataframe-expectations/commit/5447db199d7d883234d243dee029b6444e38d64f))
* added more badges to readme ([6c1b0bf](https://github.com/getyourguide/dataframe-expectations/commit/6c1b0bf4031ab5c5b66e82fe91424033452b7347))
* added publishing and release workflows ([3f89e95](https://github.com/getyourguide/dataframe-expectations/commit/3f89e950b9b2a9fdae844ad082c75e5329425722))
* added publishing and release workflows ([fd1308b](https://github.com/getyourguide/dataframe-expectations/commit/fd1308bc33f52de1b6aa5b520d9674e1e7374e9d))
* convert expectation category to str while generating stubs ([122872b](https://github.com/getyourguide/dataframe-expectations/commit/122872be9873c8145d58f2a9cf8e80cc75c478bc))
* handle pandas DataFrame.map() compatibility for older versions ([cbbf9f1](https://github.com/getyourguide/dataframe-expectations/commit/cbbf9f14acb4fd5c7f8438b81fea4297bd23284d))
* pinned action commit hashes, updated pr template ([f7b731f](https://github.com/getyourguide/dataframe-expectations/commit/f7b731fd3377086b328be12ab533522fc4cc1afb))
* update sanity_checks scripts to accomodate dynamic calls to expectation functions ([4d80f7a](https://github.com/getyourguide/dataframe-expectations/commit/4d80f7a35773df4ef0d35d04806eaa02e57c7901))
* updated release-please hash to approve version ([9a0b793](https://github.com/getyourguide/dataframe-expectations/commit/9a0b7934e2342a743f84c77063b95fb0d21877f4))
* updated release-please hash to approve version ([218e560](https://github.com/getyourguide/dataframe-expectations/commit/218e56057c0f7430dc15070bb6eac62c2ff78147))


### Documentation

* update documentation ([dbbb449](https://github.com/getyourguide/dataframe-expectations/commit/dbbb4498ee2a1c2aac89814650ede8cb45bb504d))
* update expectations_autodoc.py to remove API reference button on exp. cards ([a84b644](https://github.com/getyourguide/dataframe-expectations/commit/a84b6441834dd6cfda63157978b81fc75db1c957))
* updated readme ([1e9a92c](https://github.com/getyourguide/dataframe-expectations/commit/1e9a92c61dba3718eac6360e7a02fe0d8b0fd054))

## Changelog dataframe-expectations

## Version 0.1.1
- Migrated CI/CD from self-hosted to GitHub-hosted runners
- Added automated documentation publishing to GitHub Pages
- Pinned UV setup action to specific commit hash for stability
- Improved CI workflow organization with dedicated `build-docs` job

## Version 0.1.0
- Initial commit contains all the basic functionality for the library
- Added documentation
