# Parasite.AI - Neural Network Implementation

This Repo holds the neural network and Python Flask server for the Parasite.AI project, created at MangoHacks 2019.

## Summary

A Convolutional Neural Network  identifies Malaria from 224x224 px images of blood stain samples.

### Dependencies

Run pip install with ```requirements.txt``` to install dependencies in a virutal environment.

Run the app with:

```
flask run
```

###How to use

Parasite.AI is set up in a Flask server, being treated as an API.

To communicate with a running server, send POST requests to:

```<domain>/predict?image=<image_URL>```

Built at __MangoHacks 2019__
