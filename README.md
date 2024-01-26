# InfoDenguePredict



Comparing the use of RNN and tradional (auto-regressive) methods to forecast Epidemic time-series
Documentation is available [here](http://infodenguepredict.readthedocs.io)


## Getting started

In order to run the tests you need ssh access to the InfoDengue server. Then you have to open an ssh tunnel:
```bash
ssh -f user@infodengueserver -L 5432:localhost:5432 -N
```

to install the dependencies run the following command:
```bash
sudo pip3 install -r requirements.txt
```

This project requires a GPU to run the LSTM model efficiently.



