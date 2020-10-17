# morze
Sea sculpture for morze band

## running on Windows
Create python virtual env (only once):

````
python -m venv venv
````

Activate python virtual env (if it is not active):

````
venv\Scripts\activate.bat
````

Install requirements (only once unless they change):

````  
python -m pip install -r requirements.txt
````

Display help:

````
python main.py -h
````

Run using laptop camera:

````
python main.py -camera 0
````

Run using youtube video:

````
python main.py -stream WHPEKLQID4U
````
