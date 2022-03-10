PACKAGE:=$(shell basename $(shell pwd))
PREFIX ?=
PIP ?= pip
ifeq ($(CONDA_PREFIX),)
	PREFIX=sudo -H
	PIP=pip-sirius
endif

install: uninstall
	$(PREFIX) $(PIP) install --no-deps ./
	$(PREFIX) git clean -fdX

uninstall:
	$(PREFIX) $(PIP) uninstall -y $(PACKAGE)

develop-install: develop-uninstall
	$(PIP) install --no-deps -e ./

# known issue: It will fail to uninstall scripts
#  if they were installed in develop mode
develop-uninstall:
	$(PIP) uninstall -y $(PACKAGE)
