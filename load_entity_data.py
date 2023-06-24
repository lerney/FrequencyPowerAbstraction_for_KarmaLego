import os
import scipy.io
import numpy as np


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
