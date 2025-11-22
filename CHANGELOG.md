# Changelog

## [0.5.0](https://github.com/getyourguide/dataframe-expectations/compare/v0.4.0...v0.5.0) (2025-11-22)


### Features

* add suite_result and tagging ([e36becd](https://github.com/getyourguide/dataframe-expectations/commit/e36becd82141ad38e122d1d5209a12c2d0971b46))
* add suite_result and tagging ([e0857b4](https://github.com/getyourguide/dataframe-expectations/commit/e0857b4ae906c297431b21cfafa4166653c9d413))


### Documentation

* updated documentation ([6a08bd9](https://github.com/getyourguide/dataframe-expectations/commit/6a08bd9166d10cbf252e5f9a0e50a0e7bf192602))

## [0.4.0](https://github.com/getyourguide/dataframe-expectations/compare/v0.3.0...v0.4.0) (2025-11-10)


### ⚠ BREAKING CHANGES

* ‼️ BREAKING CHANGE: Major codebase restructuring with new module organization. However, most changes are made to the internal modules.

**What changed:**
- All internal modules have been reorganized into a `core/` package
- Expectation registry simplified from three-dictionary to two-dictionary structure with O(1) lookups
- Main imports updated from `expectations_suite` to `suite`

**Migration guide:**
Update your imports to use the new module structure:
```python
# Before
from dataframe_expectations.expectations_suite import DataFrameExpectationsSuite

# After
from dataframe_expectations.suite import DataFrameExpectationsSuite
```

### Features

* restructure codebase with core/ module and explicit imports ([42a233a](https://github.com/getyourguide/dataframe-expectations/commit/42a233ade81fc2af3ce0462ab24f189d969756bd))
* restructure codebase, and registry refactoring ([111bca1](https://github.com/getyourguide/dataframe-expectations/commit/111bca10eed21631938db310c562d0a2d73e810c))
* simplified registry ([c182858](https://github.com/getyourguide/dataframe-expectations/commit/c18285874837952bf1a7af3f2d1f21613286c34f))


### Bug Fixes

* consolidate imports ([9a76467](https://github.com/getyourguide/dataframe-expectations/commit/9a76467cd63c9ba15bd4878e247aef2b631316df))
* deleted duplicate dataclass and enums from registry ([82bec0c](https://github.com/getyourguide/dataframe-expectations/commit/82bec0ce13be1e2a1bc16fb77d0aaf91edb5692f))
* deleted duplicate DataFrameExpectation code from expectations package ([d47eb8b](https://github.com/getyourguide/dataframe-expectations/commit/d47eb8be2eef84d820653f5ef07a35e44695c5a3))
* import enums from types ([fa84764](https://github.com/getyourguide/dataframe-expectations/commit/fa847643a310a27e615290567d3e11fad4344977))
* manually trigger CI for release-please PRs ([49419e6](https://github.com/getyourguide/dataframe-expectations/commit/49419e6531c9fec51a6f46f1dde20ab2e850c1db))
* manually trigger CI for release-please PRs ([9585cf5](https://github.com/getyourguide/dataframe-expectations/commit/9585cf5d75b0e90b6f94cd14cae87922055a2212))
* return correct version when package is built ([82ff343](https://github.com/getyourguide/dataframe-expectations/commit/82ff3435c6b6ea904a1a58b71eb6a890d80991d6))



## [0.3.0](https://github.com/getyourguide/dataframe-expectations/compare/v0.2.0...v0.3.0) (2025-11-09)


### ⚠ BREAKING CHANGES

* ‼️ BREAKING CHANGE: The DataFrameExpectationsSuite API has changed. Users must now call .build() before .run().

**Migration guide:**
```python
# Before
suite.run(df)

# After
runner = suite.build()
runner.run(df)
```

**New decorator feature:**
 ```python
@runner.validate
def load_data():
    return pd.read_csv(\"data.csv\")

@runner.validate(allow_none=True)
def optional_load():
    return None  # Skip validation when None
```

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
