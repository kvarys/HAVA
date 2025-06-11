# HAVA

A brief description of what this project does and who it's for.

## 🚀 Getting Started

Clone the repository:
git clone https://github.com/kvarys/HAVA.git
or
git clone git@github.com:kvarys/HAVA.git

cd HAVA

Set up the Python environment:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Train the HAVA policy with
python main.py hava 0.1 1.0 mix

or the safe/legal policy (no social norms) with
python main.py hava 0.1 1.0 safe

or the social policy (no safety / legal norms) with
python main.py hava 0.1 1.0 dd
