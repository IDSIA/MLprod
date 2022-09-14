.. Machine Learning in Production documentation master file, created by
   sphinx-quickstart on Thu Aug 25 07:19:54 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Machine Learning in Production's documentation!
==========================================================

This documentation is meant to be a complementary material describing the implementation of our
*proof-of-concept* project for the course **Machine Learning in Production** 

.. note::
    This project is a proof-of-concept, nevertheless, state-of-the-art tools and best practices has
    been followed during the development.

.. danger::
    This project is not intended to be used in a REAL production system!

This project exploit the power of containers and the concept of microservices to build a distributed system. 
We use `Docker`_ to manage the containers and *docker-compose* to start and shutdown the entire system (for additional info see :ref:`Dockerref`).

We used open source software and made the deployment as plotform independent as possible.

Setup and installation commands are provided for Linux OS only.


Installation
============

Installation and execution is managed through the given ``docker-compose`` file.

Clone this repository, then build the container images using the ``docker-compose build`` command.

To run the application, use the ``docker-compose up -d`` command. 

More details on the :ref:`Dockerref` page.


Components
==========

Following an overview of all the components available in the application.

.. toctree::
   :glob:
   
   components/*

.. toctree::
    :maxdepth: 1
    :caption: Quick start


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Docker: https://www.docker.com/
