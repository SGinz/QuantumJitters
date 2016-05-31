# LIBRARIES
    library(caret)    

# LOAD DATA
    setwd("C:/Users/steve/OneDrive/Documents/BootCamp/Kaggle")
    train = read.csv('data/training.csv')
    test = read.csv('data/test.csv')
    source('helper.R')
    train[train==-999.0] <- NA
    test[test==-999.0] <- NA
    train$PRI_jet_num <- train$PRI_jet_num
    test$PRI_jet_num <- test$PRI_jet_num
    str(train)
    weight = train$Weight
    labels = train$Label
    train <- train[, -c(1,32,33)]
    test <- test[,-1]
    
# Subsetting data for coding purposes
    subsetSize = 10000
    dataSubset = sample(x = nrow(train), size = subsetSize)
    train.sub = train[dataSubset,]
    test.sub = test[dataSubset,]
    labels.sub = labels[dataSubset]
    weight.sub = weight[dataSubset]
   
    ColNames = c("DER_mass_MMC","DER_mass_transverse_met_lep","DER_mass_vis","DER_pt_h",
                 "DER_deltaeta_jet_jet","DER_mass_jet_jet","DER_prodeta_jet_jet","DER_deltar_tau_lep",
                 "DER_pt_tot","DER_sum_pt","DER_pt_ratio_lep_tau","DER_met_phi_centrality","DER_lep_eta_centrality",
                 "PRI_tau_pt","PRI_tau_eta","PRI_tau_phi","PRI_lep_pt","PRI_lep_eta","PRI_lep_phi","PRI_met",
                 "PRI_met_phi","PRI_met_sumet","PRI_jet_num","PRI_jet_leading_pt","PRI_jet_leading_eta",
                 "PRI_jet_leading_phi","PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt")
    
    RemoveCols = c("PRI_tau_phi","PRI_lep_phi","PRI_met_phi","PRI_jet_leading_phi","PRI_jet_subleading_phi")
  
# Diagnostics
    diagResult = data.frame(rownames = ColNames, ColNames=c('Min.','Max.','1st Qu.','Median','Mean','3rd Qu.','Max.','NA\'s'))
    summary.sub = summary(train.sub)
    
    
    
    summary(train.sub)
    sapply(train.sub,sd)
    sapply(train.sub,var)
    
    
    src.table = table(labels)
    src.table
    src.table[2] / (src.table[1] + src.table[2])

# Variable Importance
    modelFit = filterVarImp(x=train.sub,y=labels.sub)
    modelFit

# Normalize to zero mean and 1 sd
    train.sub.scaled = predict(preProcess(train.sub,method = c('center','scale')),train.sub)
    train.scaled = predict(preProcess(train,method = c('center','scale')),train)
  
# Set up trainer control and expansion grid, train and predict
    ctrl = trainControl(method = 'repeatedcv',number = 2, summaryFunction = AMS_summary)
    
    useMethod = 'gbm'   
    # Others: rf Random Forest, bag Bagged, bstTree Boosted Tree, xgbTree extreme gradient boosting,
    #         glm generalized linear model, knn k Nearest Neighbors, nnet Neural Network,
    #         pcr Principal Component Analysis
    
    model = train(x = train.sub.scaled, y = labels.sub, method = useMethod, weights = weight.sub,
                      verbose = TRUE, trControl = ctrl, metric = 'AMS')
    pred = predict(model, newdata = train.sub, type = 'prob')
    
# Set up Test Prediction and Result DF
    pred.test = predict(model, newdata = test.sub, type = 'prob')
    
    sThreshold = .002
    predicted = rep('b', length(pred.test[,2]))
    predicted[pred.test[,2] >= sThreshold] = 's'

    tbl.result = table(truth = labels.sub, pred = predicted)
    misclassed = (tbl.result['b','s'] + tbl.result['s','b']) / subsetSize
    misclassed
    

# Set up submission    
    weightRank = rank(pred.test[,2], ties.method = 'random')
    
    
    
    
    
    