### setup file for python bhc module ###

import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name = "bhc",
	version = "0.0.1",
	author = "Sam Voisin & Jonathan Klus",
	author_email = "sam.voisin@duke.edu & jonathan.klus@duke.edu",
	description = "This package implements the bayesian hierarchical clustering algorithm described by K. Heller and Z. Ghahramani (2005)",
	long_description = long_description,
	long_description_content_type = "text/markdown",
	url = "https://github.com/samvoisin/STA663_Final_Project",
	packages = setuptools.find_packages(),
	classifiers = [
		"Programming Language :: Python :: 3",
		"Development Status :: 3 - Alpha",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Natural Language :: English",
		"Intended Audience :: Science/ Research",
		"Operating System :: OS Independent"
	]
)
