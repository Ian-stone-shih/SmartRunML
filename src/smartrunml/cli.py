"""Console script for smartrunml."""
import smartrunml

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for smartrunml."""
    console.print("Replace this message by putting your code into "
               "smartrunml.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
