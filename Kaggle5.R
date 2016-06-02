# LIBRARIES
    library(caret) 
    library(pROC)
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
        # train.sub.scaled = train.sub.scaled[,]
        gbm.ctrl = trainControl(method = 'repeatedcv',
                                number = 2, 
                                summaryFunction = AMS_summary)
        gbm.Grid = expand.grid(n.trees = c(100,150,200), 
                               interaction.depth = c(5,10,15), 
                               shrinkage = 0.1,
                               n.minobsinnode = 10)
        gbm.model = train(x = train.sub.scaled, y = train.labels, method = 'gbm', weights = train.weight,
                          verbose = TRUE, trControl = gbm.ctrl, tuneGrid = gbm.Grid, metric = 'AMS')
        
        gbm.model
        plot(gbm.model)
        gbm.pred = predict(gbm.model, newdata = train.sub, type = 'prob')
        
    # Test Prediction
        gbm.pred.test = predict(gbm.model, newdata = test.sub, type = 'prob')
   
        auc.labels <- ifelse(as.character(train.labels)=="s", 1, 0)
        auc = roc(auc.labels, gbm.pred[,2])
        plot(auc, print.thres=TRUE)
        
        gbm.sThreshold = .5
        gbm.predicted = rep('b', length(gbm.pred.test[,2]))
        gbm.predicted[gbm.pred.test[,2] >= gbm.sThreshold] = 's'
    
        gbm.tbl.result = table(truth = test.labels, pred = gbm.predicted)
        gbm.misclassed = (gbm.tbl.result['b','s'] + gbm.tbl.result['s','b']) / subsetSize
        gbm.misclassed

#RANDOM FOREST MODEL
    # Trainer control and expansion grid, train and predict
        rf.ctrl = trainControl(method = 'repeatedcv',
                               number = 5, 
                               summaryFunction = AMS_summary)
        rf.Grid = expand.grid(mtry = c(3,6,9))
        rf.model = train(x = train.sub.scaled, y=train.labels, method = 'rf', weight = train.weight,
                         verbose = TRUE, trControl = rf.ctrl, tuneGrid = rf.Grid, metric = 'AMS')

        rf.model
        plot(rf.model)
    # Test Prediction
        rf.pred.test = predict(rf.model, newdata = test.sub, type = 'prob')
        
        auc = roc(auc.labels, gbm.pred[,2])
        plot(auc, print.thres=TRUE)
        
        rf.sThreshold = .0001
        rf.predicted = rep('b', length(rf.pred.test[,2]))
        rf.predicted[rf.pred.test[,2] >= rf.sThreshold] = 's'
        
        rf.tbl.result = table(truth = test.labels, pred = gbm.predicted)
        rf.misclassed = (tbl.result['b','s'] + tbl.result['s','b']) / subsetSize
        print('GBM: ' + gbm.misclassed)
        print('RF: ' + rf.misclassed)
    
        
# Neural Networks Model
    # Control, Grid, Train
        #nnet, elm, avNNet, mlp, pcaNNet, dnn, rbf
        nn.Grid = expand.grid(.size=c(1,5,10),.decay= .01, .bag=1)
        nn.model = train(y = train.labels, x = train.sub.scaled, method = 'avNNet', weights = train.weight,
                          linout = TRUE, trace= TRUE, maxit = 2000, tuneGrid = nn.Grid, metric = 'AMS')
        
        nn.model
        plot(nn.model)
        nn.pred = predict(nn.model, newdata = train.sub, type = 'prob')
        
        # Test Prediction
        nn.pred.test = predict(nn.model, newdata = test.sub, type = 'prob')
        
        auc = roc(auc.labels, nn.pred[,2])
        plot(auc, print.thres=TRUE)
        
        nn.sThreshold = .5
        nn.predicted = rep('b', length(nn.pred.test[,2]))
        nn.predicted[nn.pred.test[,2] >= nn.sThreshold] = 's'
        
        nn.tbl.result = table(truth = test.labels, pred = nn.predicted)
        nn.misclassed = (nn.tbl.result['b','s'] + nn.tbl.result['s','b']) / subsetSize
        nn.misclassed
        
# Test Models against Kaggle Test Set ---->
    test = read.csv('data/test.csv') 
    test[test==-999.0] <- NA
    test$PRI_jet_num <- test$PRI_jet_num
    test <- test[,-1]
    
    gbm.kaggle.test = predict(gbm.model, newdata = test, type= 'prob')
    gbm.kaggle.pred = rep('b',length(gbm.kaggle.test[,2]))
    gbm.kaggle.pred[gbm.kaggle.test[,2]>=gbm.sThreshold] = 's'

    rf.kaggle.test = predict(rf.model, newdata = test, type = 'prob')
    
    
    
    
    #RUN XGBOOST Nodes 30,50,1
    
# Set up submission  
    usePredModel = gbm.pred.test
    
    final.predicted <- rep("b",550000)
    final.predicted[usePredModel[,2]>=threshold] <- "s"
    final.weightRank = rank(usePredModel[,2], ties.method= "random")
    
    submission = data.frame(EventId = testId, RankOrder = final.weightRank, Class = final.predicted)
    
    write.csv(submission, "gbm_submission.csv", row.names=FALSE)
    
    
    
    
    