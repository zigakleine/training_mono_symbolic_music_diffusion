import pickle


val_losses_path = "./lakh_val_losses.pkl"
val_losses = pickle.load(open(val_losses_path, "rb"))
print(val_losses.index(min(val_losses)))