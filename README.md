# CountryCrab 🦀 

## Setting up the Jupyter Docker container

The `dockerctl` script provides options to control building and running the image.
Usage is demonstrated below:

```sh
cd docker

# this builds an image tagged `camsat`
./dockerctl build camsat

# start Jupyter lab on a port of your choice
./dockerctl run camsat <port>
```

The terminal output will then instruct you how to connect to the running instance.

### Installing CountryCrab

After running Docker to install the CountryCrab package run
```sh
pip install -e .
```
