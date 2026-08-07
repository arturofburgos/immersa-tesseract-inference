"""Main entrypoint for Immersa Tesseract Inference pipeline."""

import typer

app = typer.Typer(name="immersa_tesseract_inference")


@app.command()
def run() -> None:
    """Run the Immersa Tesseract Inference pipeline."""
    # Chain your Tesseracts here. For example, once you have built a component
    # (`make new <mytess>` then `make build <mytess>`):
    #
    #     from tesseract_core import Tesseract
    #
    #     with Tesseract.from_image("immersa_tesseract_inference_<mytess>") as tess:
    #         result = tess.apply({"example_input": ...})
    #     typer.echo(result)
    #
    # See app/chain.ipynb for an interactive version.
    typer.echo("Running Immersa Tesseract Inference pipeline...")


def entrypoint() -> None:
    """CLI entrypoint for the application."""
    app()


if __name__ == "__main__":
    entrypoint()
