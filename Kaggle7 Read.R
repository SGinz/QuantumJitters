library(caret)
source('helper.R')

# LOAD DATA
    src.train = readRDS('MichaelsData.rds')
    
    src.EventID = src.train$EventID
    src.label = src.train$Label
    src.weight = src.train$Weight
    mod.train = src.train[,c(-1,-28, -29)]
    
# SCALE AND SUBSET
    scaled.train <- predict(preProcess(mod.train, method = c('center','scale')), mod.train)
    set.seed(4)
    dataSubset <- sample(x = nrow(scaled.train), size = nrow(scaled.train) * .1)
    sub.train <- mod.train[dataSubset,]
    sub.label <- src.label[dataSubset]
    sub.weight <- src.weight[dataSubset]
    
# SET UP TRAINING MODEL
    gbm.ctrl <- trainControl(method = 'repeatedcv',
                        number = 5,
                        summaryFunction = AMS_summary)
    gbm.grid <- expand.grid(n.trees = c(50,100,150),
                       interaction.depth = c(1,5,9),
                       shrinkage = c(.01,.1,.5),
                       n.minobsinnode = c(10,100,1000))
    gbm.model <- train(x = sub.train, y = sub.label, method = 'gbm', weights = sub.weight, 
                       verbose = TRUE, trControl = gbm.ctrl, tuneGrid = gbm.grid, metric = 'AMS')
    
    gbm.model
