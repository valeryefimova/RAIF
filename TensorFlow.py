import tensorflow as tf
import pandas as pd


def read_data(num_epochs, shuffle):
    df_data = pd.read_csv(
        tf.gfile.Open("data/X.csv"),
        skipinitialspace=True,
        engine="python",
        skiprows=1)

    labels = pd.read_csv(
        tf.gfile.Open("data/y.csv"),
        skipinitialspace=True,
        engine="python",
        skiprows=1)

    return tf.estimator.inputs.pandas_input_fn(
        x=df_data,
        y=labels,
        batch_size=100,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=5)


COLUMNS = ["claim_id", "Event_type", "Period_EvCl", "Period_StEv", "Policy_agent_cat", "Owner_type", "FLAG_Owner_bl",
           "Insurer_type", "FLAG_Insurer_bl", "Policy_KBM", "Policy_KS", "Policy_KT", "Policy_KVS", "FLAG_Policy_KO",
           "FLAG_Policy_KP", "FLAG_Policy_KPR", "FLAG_Policy_type", "VEH_age", "VEH_aim_use", "VEH_capacity_type",
           "VEH_model", "VEH_type_name", "FLAG_bad_region", "FLAG_dsago", "FLAG_prolong", "Owner_region",
           "Sales_channel", "Policy_loss_count", "Damage_count", "bad", "Claim_type"]


m = tf.estimator.LinearClassifier(feature_columns=COLUMNS)
m.train(input_fn=read_data(2, True), steps=2000)

