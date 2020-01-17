# Useful commands for xarray 

First of all, let's get things straight. I absolutely love [xarray](http://xarray.pydata.org/en/stable/).  
Now that being said, I've had my fair share of late nights pulling out my hair in sheer frustration over things such as MultiIndex, data manipulations, adding new features/dims/coordinates - you name it! And it is my experience that I am not alone in this situation. 

Therefore, I decided to create this post to show some examples of some of the most efficient methods I've found to overcome some of these problems, which I feel is currently lacking in the documentation of `xarray`.  


**Let's get to it!**

1. TOC
{:toc}


## Introduction
To make this walkthrough as concrete as possible - I have decided to include a specific dataset that you might use while following along. Of course you are welcome to use your own, but obviously variable names, projection parameters, etc.- might be different.  

I generally recommend using NetCDF as much as possible, but using grib format is also possible with [cfgrib](https://github.com/ecmwf/cfgrib) by setting `engine='cfgrib'` in `xarray.open_dataarray()` command.


## Data Reading and Writing
```python
import xarray as xr

# Read NetCDF dataset 
data = xr.open_dataset('filename.nc')

```

## Data manipulation, projections, etc.
$$
\sum_n (x)
$$
