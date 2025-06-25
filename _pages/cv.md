---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* Ph.D in Artificial Intelligence, University Of Science And Technology Of China, 2029 (expected)
* B.S. in Electronic Information Science and Technology, HeFei University of Technology, 2025

Work experience
======
* Spring 2024: Academic Pages Collaborator
  * GitHub University
  * Duties includes: Updates and improvements to template
  * Supervisor: The Users
  
Skills
======
* Programming Languages
  * Python (Proficient)
  * C++ (Familiar)
  * MATLAB

* AI & Data Science Frameworks
  * PyTorch
  * Scikit-learn, Pandas, NumPy
  * OpenCV

* Developer Tools & Platforms
  * Git & GitHub
  * Docker
  * Linux / Bash Scripting
  * LaTeX

* Technical Expertise
  * Machine Learning / Deep Learning
  * Control Theory
  * Industrial Data Analysis
  * Time-Series Forecasting

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  

