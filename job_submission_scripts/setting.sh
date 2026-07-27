#!/bin/bash

export http_proxy="http://proxy.ccs.ornl.gov:3128/"
export https_proxy="http://proxy.ccs.ornl.gov:3128/"
export no_proxy="localhost,127.0.0.0/8,*.ccs.ornl.gov"

export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
export NO_PROXY="${no_proxy}"
