# training loop with adaptive lamda (learning rate)

# -----------------------------------------------------------------------------------
# Reinit weigts and the corresponding optimizer
# -----------------------------------------------------------------------------------
model = init_weights(conf, model)
opt, scheduler = init_opt(conf, model)
# Adaptive lamda adjustment
# -----------------------------------------------------------------------------------
lamda = 1.5  # Initialize lamda to a larger value
patience = 3  # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_since_improvement = 0
minimum_lamda = 1e-6  # Minimum value for lamda

# -----------------------------------------------------------------------------------
# train the model
# -----------------------------------------------------------------------------------
for epoch in range(conf.epochs):
    print(25*"<>")
    print(50*"|")
    print(25*"<>")
    print('Epoch:', epoch)

    # ------------------------------------------------------------------------
    # train step, log the accuracy and loss
    # ------------------------------------------------------------------------
    train_data = train.train_step(conf, model, opt, train_loader)

    # update history
    for key in tracked:
        if key in train_data:
            var_list = train_hist.setdefault(key, [])
            var_list.append(train_data[key])           

    # ------------------------------------------------------------------------
    # validation step
    val_data = train.validation_step(conf, model, opt, valid_loader)
    val_loss = val_data['loss']

    # update validation history
    for key in tracked:
        if key in val_data:
            var = val_data[key]
            if isinstance(var, list):
                for i, var_loc in enumerate(var):
                    key_loc = key+"_" + str(i)
                    var_list = val_hist.setdefault(key_loc, [])
                    val_hist[key_loc].append(var_loc)
            else:
                var_list = val_hist.setdefault(key, [])
                var_list.append(var)   


    scheduler.step(train_data['loss'])
    print("Learning rate:",opt.param_groups[0]['lr'])
    best_model(train_data['acc'], val_data['acc'], model=model)

        # Adaptive λ (alpha) adjustment for convolutional layers only.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1

    if epochs_since_improvement > patience:
        lamda = lamda / 2.0  # Reduce λ: this will allow more singular values (higher rank) later on
        print(f"Reducing conv layer lamda (α) to {lamda}")

        # Loop through optimizer parameter groups and update the nuclear norm regulariser for conv layers.
        for param_group in opt.param_groups:
            if 'reg' in param_group and (isinstance(param_group['reg'], reg.reg_nuclear_conv) or 
                                        isinstance(param_group['reg'], reg.reg_nuclear_linear)):
                param_group['reg'].lamda = lamda  # Update lambda for both types
        epochs_since_improvement = 0

    lamda = max(lamda, minimum_lamda)  # Prevent λ from going to zero
    conf.lamda_0 = lamda  # Update the global config value

    lin_rank_ratio = maf.linear_effective_rank_ratio(model, epsilon=1e-3)
    print(f'Average Fully Connected Layer Rank (ε=1e-3): {lin_rank_ratio}')