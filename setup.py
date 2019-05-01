### setup file for python bhc module ###

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
	long_description = fh.read()

with open("LICENSE") as lic:
	license = lic.read()

setup(
	name = "bhc",
	version = "0.0.1",
	author = "Sam Voisin & Jonathan Klus",
	author_email = "sam.voisin@duke.edu",
	description = "This package implements the bayesian hierarchical clustering algorithm described by K. Heller and Z. Ghahramani (2005)",
	long_description = long_description,
	long_description_content_type = "text/markdown",
	license = license,
	url = "https://github.com/samvoisin/STA663_Final_Project",
	packages = find_packages(),
	classifiers = [
		"Programming Language :: Python :: 3",
		"Development Status :: 3 - Alpha",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Natural Language :: English",
		"Operating System :: OS Independent"
	]
)
