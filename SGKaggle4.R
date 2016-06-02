# LIBRARIES
    library(caret) 
    source('helper.R')
# LOAD DATA
    setwd("C:/Users/steve/OneDrive/Documents/BootCamp/Kaggle")
    train = read.csv('data/training.csv')
    train[train==-999.0] <- NA
    train$PRI_jet_num <- train$PRI_jet_num
    weight = train$Weight
    labels = train$Label
    train <- train[, -c(1,32,33)]
    
    ColNames = c("DER_mass_MMC","DER_mass_transverse_met_lep","DER_mass_vis","DER_pt_h",
                 "DER_deltaeta_jet_jet","DER_mass_jet_jet","DER_prodeta_jet_jet","DER_deltar_tau_lep",
                 "DER_pt_tot","DER_sum_pt","DER_pt_ratio_lep_tau","DER_met_phi_centrality","DER_lep_eta_centrality",
                 "PRI_tau_pt","PRI_tau_eta","PRI_tau_phi","PRI_lep_pt","PRI_lep_eta","PRI_lep_phi","PRI_met",
                 "PRI_met_phi","PRI_met_sumet","PRI_jet_num","PRI_jet_leading_pt","PRI_jet_leading_eta",
                 "PRI_jet_leading_phi","PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt")
    
    RemoveCols = c("PRI_tau_phi","PRI_lep_phi","PRI_met_phi","PRI_jet_leading_phi","PRI_jet_subleading_phi")
    
# Center and Scale
    train.scaled = predict(preProcess(train,method = c('center','scale')),train)
    
# Subsetting data for coding purposes
    set.seed(4)
    subsetSize = nrow(train) * .8
    dataSubset = sample(x = nrow(train), size = subsetSize)
    
    train.sub = train[dataSubset,]
    train.labels = labels[dataSubset]
    train.weight = weight[dataSubset]
    
    train.sub.scaled = train.scaled[dataSubset,]
    
    test.sub = train[-dataSubset,]
    test.labels = labels[-dataSubset]
    test.weight = weight[-dataSubset]
    test.scaled = train.scaled[-dataSubset,]

# Diagnostics
    str(train)
    src.table = table(labels)
    src.table
    src.table[2] / (src.table[1] + src.table[2])

#GBM MODEL
    # Trainer control and expansion grid, train and predict
        gbm.ctrl = trainControl(method = 'repeatedcv',number = 5, summaryFunction = AMS_summary)
        gbm.Grid = expand.grid(interaction.depth = c(1, 5, 9), shrinkage = 0.1,n.minobsinnode = 10)
        gbm.model = train(x = train.sub.scaled, y = train.labels, method = 'gbm', weights = train.weight,
                          verbose = TRUE, trControl = gbm.ctrl, tuneGrid = gbm.Grid, metric = 'AMS')
        
        gbm.model
        plot(gbm.model)
        gbm.pred = predict(gbm.model, newdata = train.sub, type = 'prob')
        
    # Test Prediction
        gbm.pred.test = predict(gbm.model, newdata = test.sub, type = 'prob')
        
        sThreshold = .002
        gbm.predicted = rep('b', length(gbm.pred.test[,2]))
        gbm.predicted[gbm.pred.test[,2] >= sThreshold] = 's'
    
        tbl.result = table(truth = test.labels, pred = gbm.predicted)
        misclassed = (tbl.result['b','s'] + tbl.result['s','b']) / subsetSize
        misclassed

#RANDOM FOREST MODEL
    rf.ctrl = trainControl()
    rf.Grid = expand.grid()
    rf.model = train(x = train.sub.scaled, y=train.labels, method = 'rf', weight = train.weight,
                     verbose = TRUE, trControl = rf.ctrl, tuneGrid = rf.grid, metric = 'AMS')

    rf.model
    plot(rf.model)
    
    
    
# Kaggle Test Set ---->
    test = read.csv('data/test.csv') 
    test[test==-999.0] <- NA
    test$PRI_jet_num <- test$PRI_jet_num
    test <- test[,-1]
    
    
    
# Set up submission    
    weightRank = rank(pred.test[,2], ties.method = 'random')
    
    
    
    
    
    