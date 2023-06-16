import os
import scipy.io
import json
import argparse
import numpy as np
import itertools
import scipy.stats as stats
from scipy.integrate import simps
from scipy import signal
import traceback
import pickle



def load_entity_data(folder_path):
    entity_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".mat"):
            file_path = os.path.join(folder_path, file_name)
            mat_data = scipy.io.loadmat(file_path)

            session = mat_data['EEG']['session'][0][0][0]
            subject = mat_data['EEG']['subject'][0][0][0]
            data = mat_data['EEG']['data'][0][0].transpose(0, 2, 1)
            data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
            entity_name = f"{subject}-{session}"

            entity_data.append({
                "entity_name": entity_name,
                "subject": subject,
                "session": session,
                "data": data
            })

    channel_names = np.concatenate(mat_data['EEG']['chanlocs'][0][0][0]['labels'])
    return entity_data, channel_names


def calculate_intervals(entity_data, interval_length,sample_rate):
    for entity_info in entity_data:
        entity_info['intervals'] = []
        total_duration = entity_info['data'].shape[1] / sample_rate
        num_intervals = int(total_duration / (interval_length/1000))
        for i in range(num_intervals):
            start_time = float(i * interval_length/1000)
            end_time = float((i + 1) * interval_length/1000)
            interval = {
                'startTime': format_time(start_time),
                'endTime': format_time(end_time),
                'bands': {}
            }
            entity_info['intervals'].append(interval)


def allocate_data(entity_data, sample_rate, interval_length):
    for entity_info in entity_data:
        num_samples = entity_info['data'].shape[1]
        num_channels = entity_info['data'].shape[0]
        num_intervals = len(entity_info['intervals'])
        samples_per_interval = int(sample_rate * interval_length/1000)
        for i in range(num_intervals):
            start_sample = i * samples_per_interval
            end_sample = start_sample + samples_per_interval
            allocated_data = entity_info['data'][:, start_sample:end_sample]
            entity_info['intervals'][i]['data'] = allocated_data
        entity_info['data'] = []


def interpolate_data_and_calculate_mean_power(entity_data, frequency_bands, sample_rate, number_of_categories, interval_length):

    num_channels = np.size(entity_data[0]['intervals'][0]['data'], 0)
    levels = {band_name: {channel: {}
                          for channel in range(num_channels)}
              for band_name in frequency_bands}
    mean_powers = {}
    # Initialize the mean powers for each band and channel
    for band in frequency_bands:
        mean_powers[band] = {}
        for channel in range(num_channels):
            mean_powers[band][channel] = []

    for entity_info in entity_data:
        num_intervals = len(entity_info['intervals'])
        for i in range(num_intervals):
            allocated_data = entity_info['intervals'][i]['data']
            num_channels = allocated_data.shape[0]
            num_samples = allocated_data.shape[1]
            for channel in range(num_channels):
                channel_data = np.append(allocated_data[channel], np.zeros(sample_rate-num_samples))
                for band_name, [low_freq, high_freq] in frequency_bands.items():

                    # Calculate the power
                    freq, psd = calculate_power(channel_data, sample_rate)

                    # Extract the power within the frequency band of interest
                    band_mask = (freq >= low_freq) & (freq <= high_freq)

                    # Frequency resolution
                    freq_res = freq[1] - freq[0]

                    # Compute the absolute power by approximating the area under the curve
                    mean_power = simps(psd[band_mask], dx=freq_res)
                    mean_powers[band_name][channel].append(mean_power)

                    # Check if the key exists in the bands dictionary and create it if not
                    if band_name not in entity_info['intervals'][i]['bands']:
                        entity_info['intervals'][i]['bands'][band_name] = {}
                    entity_info['intervals'][i]['bands'][band_name][channel] = {'mean_power': mean_power}
            entity_info['intervals'][i]['data'] = []

    # Calculate levels based on mean power values across all entities
    for band_name in frequency_bands:

        # Calculate power for each channel within the frequency band
        for channel in range(num_channels):
            mean_power_avg = np.mean(mean_powers[band_name][channel])
            mean_power_std = np.std(mean_powers[band_name][channel])

            # Calculate percentiles based on normal distribution
            percentiles = np.linspace(0, 100, number_of_categories+2)[1:-1]
            level_values = stats.norm.ppf(percentiles / 100, loc=mean_power_avg, scale=mean_power_std)

            levels[band_name][channel]['level_values'] = level_values

    return levels


def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"


def calculate_power(data, sample_rate):
    win = len(data)
    freq, psd = signal.welch(data, sample_rate, nperseg=win)

    return freq, psd


def assign_labels(entity_data, levels, channel_names):
    for entity_info in entity_data:
        num_intervals = len(entity_info['intervals'])
        for i in range(num_intervals):
            interval_data = entity_info['intervals'][i]
            for band_name, band_info in interval_data['bands'].items():
                for channel_idx, channel_name in enumerate(channel_names):
                    mean_power = band_info[channel_idx]['mean_power']

                    # Find the closest level based on mean power
                    closest_level = min(levels[band_name][channel_idx]['level_values'], key=lambda x: abs(mean_power - x))

                    # Get the number of the closest level
                    level_number = np.where(levels[band_name][channel_idx]['level_values'] == closest_level)

                    # Assign the label with the desired structure
                    label = f"{channel_name}-{band_name}-L{level_number[0][0]}"

                    # Assign the label to the corresponding channel in the entity info
                    interval_data['bands'][band_name][channel_idx]['label'] = label

    return entity_data


def generate_abstracted_features(channel_names, frequency_bands, levels_to_save):
    abstracted_features = []

    # Generate combinations of channel names, frequency bands, and number of categories
    combinations = list(itertools.product(channel_names, frequency_bands.keys(), np.array(levels_to_save)))

    # Format the combinations as abstracted features
    for channel, band, level in combinations:
        feature = f"{channel}-{band}-L{level}"
        abstracted_features.append(feature)

    return abstracted_features


def save_abstracted_features(entity_data, channel_names, frequency_bands,
                             number_of_categories, output_folder, levels_to_save):
    abstracted_features = generate_abstracted_features(channel_names, frequency_bands, levels_to_save)

    for entity_info in entity_data:
        entity_data_records = {"entityName": entity_info['entity_name'],"class": 1,"records": []}

        for interval_info in entity_info['intervals']:
            start_time = interval_info['startTime']
            end_time = interval_info['endTime']
            for band_name, band_info in interval_info['bands'].items():
                for channel_idx, channel_name in enumerate(channel_names):
                    label = interval_info['bands'][band_name][channel_idx]['label']
                    if int(label[-1]) in np.array(levels_to_save):
                        # Create a record with start time, end time, and label
                        record = {"startTime": start_time,"endTime": end_time,"label": label}

                        # Add the record to the entity's data
                        entity_data_records['records'].append(record)

        # Generate the output file name
        output_file = f"{output_folder}\\{entity_info['entity_name']}_freq_power_abs.json"

        # Create the final JSON object
        output_data = {
            "abstractedFeatures": abstracted_features,
            "entitiesData": [entity_data_records]
        }

        # Save the output JSON to a file
        with open(output_file, 'w') as f:
            json.dump(output_data, f)


def read_config(config_file_path):
    with open(config_file_path, "r") as f:
        config = json.load(f)
    return config


def save_workspace(workspace, filename):
    with open(filename, 'wb') as file:
        pickle.dump(workspace, file)


def load_workspace(filename):
    with open(filename, 'rb') as file:
        workspace = pickle.load(file)
    return workspace


def main(config_file_path):
    try:

        config = read_config(config_file_path)

        sample_rate = config["sample_rate"]
        mats_folder_path = config["mats_folder_path"]
        interval_length = config["interval_length_ms"]
        number_of_categories = config["number_of_categories"]
        frequency_bands = config["frequency_bands"]
        output_folder = config["output_folder"]
        direct_to_step = config["direct_to_step"]
        levels_to_save = config["levels_to_save"]

        print("Sample Rate:", sample_rate)
        print("MATs Folder Path:", mats_folder_path)
        print("Interval Length in mS:", interval_length)
        print("Number of Categories:", number_of_categories)
        print("Frequency Bands", frequency_bands)
        print("Output Folder:", output_folder)
        print("Direct To Step:", direct_to_step)
        print("Levels To Save:", levels_to_save)

        if direct_to_step != 'True':

            # Step1: Load entity data
            entity_data, channel_names = load_entity_data(config['mats_folder_path'])
            print("Finished Step 1: Load entity data")

            # Step 2: Calculate intervals
            calculate_intervals(entity_data, interval_length, sample_rate)
            print("Finished Step 2: Calculate intervals")

            # Step 3: Allocate data
            allocate_data(entity_data, sample_rate, interval_length)
            print("Finished Step 3: Allocate data")

            # Step 4: Calculate mean power
            levels = interpolate_data_and_calculate_mean_power(
                entity_data, frequency_bands, sample_rate, number_of_categories, interval_length)
            print("Finished Step 4: Interpolate and Calculate mean power")

        else:
            entity_data = load_workspace('entity_data.pkl')
            channel_names = load_workspace('channel_names.pkl')
            levels = load_workspace('levels.pkl')

        # Step 5: Assign labels
        entity_data = assign_labels(entity_data, levels, channel_names)
        print("Finished Step 5: Assign labels")

        # Step 6: Save abstracted features
        save_abstracted_features(entity_data, channel_names, frequency_bands,
                                 number_of_categories, output_folder, levels_to_save)
        print("Finished Step 6: Save abstracted features")
        print("Great Success!!!")

    except Exception as e:
        # Print the error message
        print("An error occurred:", str(e))

        # Print the traceback
        traceback.print_exc()

        # Save the workspace

        save_workspace(entity_data, 'entity_data.pkl')
        save_workspace(channel_names, 'channel_names.pkl')
        save_workspace(levels, 'levels.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script Configuration")
    parser.add_argument("--config", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    if args.config:
        main(args.config)
    else:
        print("Please specify a configuration file using --config argument.")



