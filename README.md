# STUDY OF IMPLEMENTATIONS OF MERGESORT ALGORITHM

Implementation of short version of Mergesort algorithm using CUDA Framework. It contains two versions for the algorithm: 
one of them developed using CUDA and the other using C language. It also includes the time of execeution of each version for 
creating a map of performance. 

## PROJECT DEVELOPMENT

This project has been designed basing in a performance study of process to sort arrays with dimensions: 1, 10, 100 and 
1000 of random values using Mergesort algorithm. It includes two version of the algorithm (sequential and parallel computing) 
and both of them are coding in C. 
It also includes different tools for monitoring the execution time, used for creating different performance maps, which 
allows calculating the speed up of each case (for 1, 10, 100 and 1000 random values)
## RESULTS 

By the use a computer eqquiped with a NVIDIA GTX 780ti graphic card and processor AMD FX-8350 obtained the following results: 
```
 PARALLEL IMPLEMENTATION: (time in milliseconds)
      Array with 1 element: 0.80
      Array with 10 elements: 0.36
      Array with 100 elements: 0.48
      Array with 1000 elements: 4.17
  
  SEQUENTIAL IMPLEMENTATION: (time in milliseconds)
      Array with 1 element: 2.00
      Array with 10 elements: 17.20
      Array with 100 elements: 52.64
      Array with 1000 elements: 319.52
```

## CONCLUSIONS 
 
 Using parallel computing instead of sequential, the different speed-up to sort arrays with different dimensions are: 
 ```
     Array with 1 element: 2.50
     Array with 10 elements: 47.77
     Array with 100 elements: 109.66
     Array with 1000 elements: 79.62
```
     
 *Using the speedup formula = secuential_time / parallel_time
 
 ## AUTHOR

* **Daniel Lopez Bustillo** - *PARALLEL PERFORMING* - (https://github.com/DanielLBustillo)
