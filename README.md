### Seismic Wave Detection on Moon and Mars

This project focuses on detecting seismic waves from extraterrestrial environments, such as the Moon and Mars, using real velocity data. By applying time-series feature engineering and training a machine learning model (Random Forest), the system can detect potential seismic events with visual and statistical outputs.

Dataset Description

The dataset contains high-frequency seismic readings (e.g., velocity vs. absolute time) captured from instruments like:

* InSight Lander (Mars)
* Chandrayaan-2 OHRC/Seismic payload (Moon)
* Other mission-generated CSV files with time-stamped velocity measurements

    Each record includes:

`abs_time`: Absolute timestamp of the reading
`velocity`: Measured ground velocity
`rel_time`: Relative time in seconds (optional/derived)

Data Format: CSV
Size Used: First 1000–10000 samples for rapid prototyping and visualization
Preprocessing: NaNs removed, velocity difference computed, rolling features added

Key Features

* Seismic Event Labeling: using velocity change threshold (95th percentile)
* Random Forest Classifier: to learn patterns in engineered features
* Feature Engineering: velocity difference, rolling mean, rolling std
* Real-time Plotting: velocity vs. time with seismic events highlighted
* Evaluation: Classification report & accuracy score

Tech Stack

* Languages: Python
* Libraries: NumPy, Pandas, scikit-learn, Matplotlib
* API: FastAPI (for future web deployment)

Use Cases

* Autonomous detection of moonquakes or marsquakes
* Supporting planetary geology research
* Enhancing rover autonomy in unstable terrain
* Pre-screening seismic events before scientific review

Future Work

* Add RNN-based models (LSTM/GRU) for sequential prediction
* Expand to multi-axis seismic data (X, Y, Z components)
* Automate alert system for major seismic anomalies
