# WISD Hackathon Submission

This project is part of my submission to the WISD hackathon. This year, we were given anonymized data from a 3-game series between two MLB affiliate teams. The data consisted of at-bat level information, with positional data tracking the path of the ball and bat.

## Interactive Dashboard
[The Webapp](https://wisd-hackathon-bat-tracking-dashboard.streamlit.app/)

I chose to create an interactive dashboard for MLB coaching staff to explore each at-bat event. The web app, hosted with Streamlit, has three main pages:

1. **At-Bat Page**: Allows users to select a player and specific event to analyze the swing. Users are presented with various values such as launch angle, exit velocity, attack angle, and more to analyze the swing.
2. **Compare Players Page**: Enables users to explore and compare two different batters' swings.
3. **Calculations Explained Page**: Explains how many of the calculations were done, given only time and positional data.

## Motivation

The motivation for the dashboard came from various sources. Firstly, the articles on Driveline Baseball were very helpful in understanding bat swing and mechanics. Specifically, inspiration was taken from [this article](https://www.drivelinebaseball.com/2022/12/hitting-biomechanics-barrel-direction/). MLB’s Baseball Savant was also used for further understanding baseball metrics and analysis tools. Additionally, I leveraged ChatGPT to assist in developing the frontend of the web app, addressing some technical issues initially deploying with Streamlit.

## Future Extensions

For future extensions, it would be interesting to determine where (x, y, z) the batter was expecting the ball to be at the time of the hit versus where the ball actually was. Comparing the launch angle, attack angle, and exit velocity of the ball at these two points would be a good way to measure the swing quality and the batter’s swing accuracy, identifying how close they were to their initial prediction.
