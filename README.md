# FrequencyPowerAbstraction_for_KarmaLego
The code purpose is to create an ordinal categorical levels abstraction for each 
electrode (channel) and frequency band in constant time intervals. The calculation is
made by using interpolation and FFT, calculation of mean power for each channel, 
band and time intervals. Afterwards calculate the mean of mean powers and std of 
mean powers for each channel and band and create the levels based on percentage 
of normal-distributed data with these mean and std, along with number of levels 
(categories). After the calculation of levels, the code assign levels for each interval, 
frequency band and channel (combinations) and saves it under Json file ready for 
single KarmaLego usage (one file for each entity – EEG record).

look inside the resp there are 2 extended manuals in pdf
