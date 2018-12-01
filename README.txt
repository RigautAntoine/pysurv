## Description

This is my attempt at building a Python package for estimating and testing survival models.

There are three types of models currently implemented:
* Nonparametric: Kaplan-Meier and Nelson-Aalen
* Semi-parametric: Cox Proportional Hazards
* Parametric, also called Accelerated Failure Time models: Weibull, Exponential and Log-Logistic models

In addition, the package has an implementation of the Aaelen Additive model. 

## To be implemented

* Gradient estimates and Newton-Raphson method for fitting AFTs and Cox. Current implementation leverages Scipy optimization on a likelihood function. 
* Log-log plots and other diagnostic tools for semi-parametric and parametric models (e.g. testing the proportionality assumption on a Cox model)
* Handling other types of censoring - primarily left- and interval-censoring
* Competing risks models
