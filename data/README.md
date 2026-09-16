# Data directory

This project uses the [UCI Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

## Expected layout

```text
data/
├── raw/
│   └── household_power_consumption.txt
├── interim/
└── processed/
```

Download and extract the source data from the repository root:

```bash
python src/download_data.py
```

Raw, interim, and processed datasets are intentionally excluded from Git because they are large and reproducible from the cited public source. Do not commit household-level readings or personally identifiable information. The notebooks create derived data in chronological order and preserve the documented 70/15/15 split.
