import pickle


# val_losses_path = "./val_losses_nes_2310.pkl"
# val_losses_path = "./val_losses_lakh_nes_2310.pkl"
val_losses_path = "./val_losses_lakh.pkl"

val_losses = pickle.load(open(val_losses_path, "rb"))
print(val_losses.index(min(val_losses)))
print(min(val_losses))