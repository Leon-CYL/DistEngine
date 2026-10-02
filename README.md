# DistEngine

## Build

Run these commands from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

The first command configures the project in `build/`. The second compiles the
`distengine_core` library and the `test_version` executable.

## Run

The current executable is the version test. On macOS or Linux, run it directly:

```bash
./build/test_version
```

To run all registered tests and display output if a test fails:

```bash
ctest --test-dir build --output-on-failure
```

After editing C++ files, rebuild and rerun the tests:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
