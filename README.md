# penneyproject

To get this project working, you can run these commands:
```bash
uv build && uv install && uv run main.py
```

Alternatively, if above doesn't work, then to compile the parser, assuming your cwd is this root, do this command, restart your notebook, and it should work from there:
```bash
cd src && python3 setup.py build_ext --inplace && cd ..
```