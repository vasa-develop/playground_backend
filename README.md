## playground_backend

This is a backend for [playground.vasa.bio](https://playground.vasa.bio).

## Installation

1. Install the dependencies.

```bash
pip install -r requirements.txt
```

2. Install [prodigal](https://github.com/hyattpd/prodigal/wiki/Installation)

3. Start the server

```bash
uvicorn main:app --reload
```