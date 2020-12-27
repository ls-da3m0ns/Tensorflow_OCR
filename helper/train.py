def train(model,epochs,batch_size,dataset,chkpt_dir):
    print(f"Training Model for {epochs} epochs with batch size of {batch_size} and checkpoint directory is {chkpt_dir}")
    for epoch in range(epochs):
        iteration = 0
        while True:
            trainX,trainY,terminator = dataset.next_train_batch(batch_size=batch_size)
            testX,testY,terminator_test = dataset.next_test_batch(batch_size=batch_size)
            temp = model.fit(trainX,trainY, epochs=1, validation_data=(testX,testY),verbose=0)
            iteration += 1

            if terminator is True:
                break
            
            if iteration/100 == 0:
                print(f"{iteration} iterations compeleted")
                
        print(f"Epoch no {epoch} no of iteration {iteration} perfomance on full testset {model.evaluate(dataset.get_full_test())}")


