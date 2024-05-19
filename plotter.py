import matplotlib.pyplot as plt

def plot_accuracies(epochs, train_accuracies, val_accuracies, labels):
    """
    Plots training and validation accuracies over epochs.

    Parameters:
    epochs (list or array): Array of epoch numbers.
    train_accuracies (list of lists or arrays): List of training accuracy arrays.
    val_accuracies (list of lists or arrays): List of validation accuracy arrays.
    labels (list): List of labels for each set of accuracies.
    """
    plt.figure(figsize=(12, 6))

    # Plot training accuracies
    plt.subplot(1, 2, 1)
    for i, train_accuracy in enumerate(train_accuracies):
        plt.plot(epochs, train_accuracy, label=labels[i])
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot validation accuracies
    plt.subplot(1, 2, 2)
    for i, val_accuracy in enumerate(val_accuracies):
        plt.plot(epochs, val_accuracy, label=labels[i])
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example data
    # Resnet - No Normalization
    epochs = [1, 2, 3, 4, 5]
    #LR2e-4
    res_no_norm_train_accuracy = [0.5893,0.6210,0.6927,0.8106,0.8849]
    res_no_norm_val_accuracy = [0.5780,.5895,0.6236,0.6137,0.6059]
    #LR1e-3
    res_norm_train_accuracy = [0.5859,0.6280,0.7008,0.8082,0.8794]
    res_norm_val_accuracy = [0.5652,0.5858,.6158, 0.6030,0.5946]
    #LR1e-3
    res_NHV_train_accuracy = [0.5782, 0.6189, 0.6420, 0.6811, 0.7037]
    res_NHV_val_accuracy = [0.5651,0.5956,0.6037,0.6188,0.6239]

    #LR1e-3
    res_NHVC_train_accuracy =  [0.5708,0.6017,0.6247,0.6419,.6550]
    res_NHVC_val_accuracy = [0.5612,0.5878,0.5946,0.6070,0.6155]
    #ViT Testss
    vit_no_norm_train_accuracy = [0.5900, 0.6426,  0.7282, 0.8552, 0.9416]
    vit_no_norm_val_accuracy = [0.5759, 0.5919,0.5999, 0.6061, 0.5942]

    vit_norm_train_accuracy = [0.6105,.6651,0.7589,0.9090, 0.9688]
    vit_norm_val_accuracy = [0.5889, 0.6094,0.6093,0.5926,0.5854]

    vit_NHV_train_accuracy = [0.5918,.6333,0.6815,0.7453,.8021]
    vit_NHV_val_accuracy = [0.5817,0.6114,0.6265,0.6279,.6297]

    vit_NHVC_train_accuracy = [0.6093,0.6351,0.6722,0.7223,0.7623]
    vit_NHVC_val_accuracy = [0.6012,0.6110,0.6306,0.6370,0.6360]

    vit_NHVC_weightdecay_0_1_train_accuracy=[0.5980,0.6184,0.6582,0.7098,0.7539]
    vit_NHVC_weightdecay_0_1_val_accuracy=[0.5897,0.5922,0.6172, 0.6229,0.6286]

    vit_NHVC_weightdecay_0_01_train_accuracy=[0.5973,0.6343,0.6699,0.7049,0.7498]
    vit_NHVC_weightdecay_0_01_val_accuracy=[0.5907,0.6160,0.6297,0.6327,0.6420]

    vit_NHVC_weightdecay_0_0001_train_accuracy=[0.5914,0.6293,0.6685, 0.7094,0.7488]
    vit_NHVC_weightdecay_0_0001_val_accuracy=[0.5855,0.6116,0.6316,0.6370,0.6452]
    train_accuracies = [
        vit_no_norm_train_accuracy,
        #vit_norm_train_accuracy,
        #vit_NHV_train_accuracy,
        vit_NHVC_train_accuracy,
        vit_NHVC_weightdecay_0_1_train_accuracy,
        vit_NHVC_weightdecay_0_01_train_accuracy,
        vit_NHVC_weightdecay_0_0001_train_accuracy
    ]

    val_accuracies = [
        vit_no_norm_val_accuracy,
        #vit_norm_val_accuracy,
        #vit_NHV_val_accuracy,
        vit_NHVC_val_accuracy,
        vit_NHVC_weightdecay_0_1_val_accuracy,
        vit_NHVC_weightdecay_0_01_val_accuracy,
        vit_NHVC_weightdecay_0_0001_val_accuracy
    ]

    # Corresponding labels
    # labels = [
    #     'ViT Baseline',
    #     #'ViT N',
    #     #'ViT NHV',
    #     'ViT NHVC',
    #     'ViT NHVC+WD 0.1',
    #     'ViT NHVC+WD 0.01',
    #     'ViT NHVC+WD 0.0001'
    # ]
    train_accuracies = [
        res_no_norm_train_accuracy,
        res_norm_train_accuracy,
        res_NHV_train_accuracy,
        res_NHVC_train_accuracy
    ]
    val_accuracies = [
        res_no_norm_val_accuracy,
        res_norm_val_accuracy,
        res_NHV_val_accuracy,
        res_NHVC_val_accuracy
    ]
        # Corresponding labels
    labels = [
        'Resnet Baseline',
        'Resnet N',
        'Resnet NHV',
        'Resnet NHVC'
    ]

    # Plot the accuracies
    plot_accuracies(epochs, train_accuracies, val_accuracies, labels)
    # Plot the training progress