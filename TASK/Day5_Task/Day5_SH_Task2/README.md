# Employee Visualisations

This project reads employee data from `employees.csv`, creates five charts, and saves them in `reports/figures/`.

## Files

- `Day5_SH_Task1.ipynb` — analysis notebook
- `employees.csv` — input data, stored beside the notebook
- `requirements.txt` — required Python packages
- `reports/figures/` — saved charts
- `reports/Day5_SH_Task1.html` — exported notebook report

## Requirements

Create `requirements.txt` with:

    pandas
    matplotlib
    jupyter
    nbconvert

## Run in the VS Code terminal

Open the project folder in VS Code. Select **Terminal → New Terminal**, then run:

    py -m pip install -r requirements.txt

To open Jupyter Notebook, run:

    py -m jupyter notebook

Open `Day5_SH_Task1.ipynb` in VS Code and run its cells from top to bottom. Keep `employees.csv` in the same folder as the notebook.

## Export the notebook as HTML

In the VS Code terminal, make sure the current directory is the folder containing `Day5_SH_Task1.ipynb`, then run:

    py -m jupyter nbconvert --to html Day5_SH_Task1.ipynb --output-dir reports

The HTML report will be saved in the `reports` folder.