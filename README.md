
# How our app works?

## Setting up
Download AiWhisper (https://github.com/openai/whisper) by importing  

> pip install git+https://github.com/openai/whisper.git 

and add the Environmental Variable to path.

Also download FFmpeg from the website https://github.com/BtbN/FFmpeg-Builds/releases this file: ffmpeg-master-latest-win64-gpl-shared.zip. Then extract it and add the bin to the Environmental Variable path.

Right now you are all set up for executing the application.

## Starting and usage

It is on localhost. Run the file. It may take a while, because we train the data model with each start.

> app.py

in root/UI and go to localhost http://127.0.0.1:5000.

Right at the start you will see a pop-up with our policy - how do we define extremist view and bad language.

When you close the pop-up you can choose a file to upload. And when you are ready click the button 'Upload' to start the algorithm. It may run for a little while, but we want it to be thorough. When it is ready it will list out all of the timestamps at which inappropriate language was use. The timestamps are in seconds.


## Licenses

In **root/app/speech_convert.py** we used:
- whisper (https://github.com/openai/whisper) - MIT License
- FFmpeg (https://ffmpeg.org/) - LGPL/GPL License

In **root/app/find_extremes.py** apart from some python libraries we used:

- https://github.com/t-davidson/hate-speech-and-offensive-language - MIT License

In **root/data/database** we used two databases:

- https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en database en.txt - Creative Commons Attribution 4.0 International License
- the database **[extremist_vs_appropriate_dataset.csv](root/data/database/extremist_vs_appropriate_dataset.csv)** was created by OpenAI ChatGPT
- Video from platform YouTube from creator **About Munich** with title **OKTOBERFEST: All the major beer tents explained - and tips on how to behave** posted on 24.09.2024 with License Creative Commons [link](https://www.youtube.com/watch?v=ZMhyG27O480)
- Video from platform YouTube from creator **The Humanist Report** with title **Anti-Gay Republicans Can’t Stop Getting Caught Up in the GAYEST Scandals Imaginable** posted on 30.09.2025 with License Creative Commons [link](https://www.youtube.com/watch?v=xCnRJvD6kIw)
- Video from platform YouTube from creator **The Mercurial Cyclist** with title **I Shortened My Cranks Like Tadej Pogačar and Wout Van Aert, Here's What Happened....** posted on 27.07.2025 with License Creative Commons [link](https://www.youtube.com/watch?v=1lxmsPbyFio)
- Video from platform YouTube from creator **Reagan Library** with title **President Reagan's Remarks Announcing the Drug Abuse Initiative on August 4, 1986** posted on 20.08.2016 with License Creative Commons [link](https://www.youtube.com/watch?v=EFtc-NXSX7Y)
- Video from platform YouTube from creator **mhmdbbn** with title **margot robbie and greta gerwig on that barbie shoe shot** posted on 30.06.2023 with License Creative Commons [link](https://www.youtube.com/watch?v=vB-2aM4gAe4)

The directory transcripts contains transcriptions of audios and videos.
The directory inputs contains uploaded media.


In **UI/static** we used:

- tuDelft.jpg picture with License Creative Commons [link](https://commons.wikimedia.org/wiki/File:TU-Delft-Bibl-2.jpg)

In **UI/templates** we used:

- Flask with BSD-3-Clause License [link](https://flask.palletsprojects.com/en/stable/)

Moreover OpenAI ChatGPT was used for inspiration.


