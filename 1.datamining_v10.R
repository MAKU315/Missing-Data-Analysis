rm(list=ls())

######################################
######### 폴더 명 경로 설정
setwd("C:/Users/korea/Desktop/dataset_kor")

install.packages("KRLS")
install.packages("Hmisc")
install.packages("extraTrees")
install.packages("glmnet")
install.packages("e1071")
install.packages("rpart")
install.packages("party")
install.packages("C50")
install.packages("CHAID")
install.packages("tree")
install.packages("randomForest")
install.packages("caret")
install.packages("kknn")
install.packages("FNN")
install.packages("class")
install.packages("cluster")
install.packages("httpuv")
install.packages("klaR")
install.packages("philentropy")
install.packages("arules")
install.packages("arulesViz")
install.packages("arulesCBA")

library("arulesCBA")
library("philentropy")
library("arules")
library("arulesViz")
library("KRLS")
library("fastDummies")
library("Hmisc")
# NEED the JAVA
options( java.parameters = "-Xmx2g" )
library("extraTrees")

# penalty regression library
library("glmnet")

# Naive Bayesian classifer & SVM library
library("e1071")

# decision tree library
library("rpart") 
library("party")
library("C50")
library("CHAID")
library("tree")

# decision tree library
library("randomForest")

# classiﬁcation and regression training
library("caret")

library("klaR")
library("MASS")

# knn
library("kknn")
library("FNN")
library("class")

# hierarchical, CLARA..  clustering.. 
library("cluster")





kor_dum_all <-read.csv("./교통사망사고정보/Kor_Train_교통사망사고정보(12.1~17.6).csv",sep=',')
result_kor <-read.csv("result_kor.csv",sep=',')


test_col <- c("주야",	"요일",	"사망자수",	"사상자수",	"중상자수",	"경상자수",	"부상신고자수",	"발생지시도",	
              "발생지시군구",	"사고유형_대분류",	"사고유형_중분류",	"법규위반",	"도로형태_대분류",	"도로형태",	"당사자종별_1당_대분류",	"당사자종별_2당_대분류")

data_kor<- kor_dum_all[,colnames(kor_dum_all) %in% test_col]
#colnames(kor_dum_all)
test_kor<-read.csv("test_kor.csv",sep=',')
test_kor[test_kor==''] <- NA


# 사고유형 대분류에서 건널목은 우선 제외
# 사상자는 맞추지 않는다
# 발생지도만 우선 맞춘다
# 사고 유형 대분류 우선 맞춘다

min.freq <- dim(kor_dum_all)[1] * 0.025 
# 0.01  250.37  23582    /25037
# 0.025  625.925 17308    /25037
# 0.05 1251.85 10175    /25037
col_list <- colnames(data_kor)
# 1 Step : 
cat_list1 <- c("주야", "요일", "사고유형_대분류","사고유형_중분류", "법규위반", "도로형태_대분류", "도로형태",
               "당사자종별_1당_대분류", "당사자종별_2당_대분류","발생지시도")
num_list1 <- c( "사망자수","중상자수","경상자수","부상신고자수")
num_list2 <- c( "사망자수","중상자수","경상자수","부상신고자수","사상자수")

anl_list1 <- c("주야", "요일", "사망자수", "중상자수", "경상자수", "부상신고자", "사고유형_대분류", "법규위반", "도로형태_대분류", 
               "당사자종별_1당_대분류", "당사자종별_2당_대분류","발생지시도")
del_list1 <-c("사상자수","도로형태","발생지시군구")
anl_list2 <- c(anl_list1,del_list1)

#### 비중요 변수 제거
# Testset의 obs의 타겟변수 서치
test_kor1 <- test_kor[,!(colnames(test_kor) %in% del_list1)]

# 더미 X
data_kor1 <- data_kor[,!(colnames(test_kor) %in% del_list1)]

pred_dat <- cbind(data_kor1, 1:dim(data_kor1)[1])
dist_dat <- cbind(data_kor1, 1:dim(data_kor1)[1])
names(dist_list) <- colnames(dist_dat)

dist_list<-list()
for(d.idx in c(1:2, 7:13)){
  
  pred_list <- names(table(data_kor1[,d.idx])[table(data_kor1[,d.idx]) >= min.freq])
  dist_list1 <- names(table(data_kor1[,d.idx])[table(data_kor1[,d.idx]) < min.freq])
  
  if(length(pred_list) > 0){
    pred_dat <- pred_dat[ (pred_dat[,d.idx] %in% pred_list) ,]
  }
  
  # make the dist list
  ifelse(length(dist_list1) > 0,
         dist_list[d.idx] <- list(dist_list1),
         dist_list[d.idx] <- NA)
}

dist_dat <- dist_dat[-pred_dat[,14],-14]
pred_dat <- pred_dat[,-14]
dist_list
#dim(pred_dat);dim(dist_dat);dim(data_kor)

# del_list1 : 마이너 변수 제거  / list : 분석할 변수 값 -> 그 외 dist_dat 로/ 
last.num <- 0 # 
result_mat <- matrix(0,nrow=sum(is.na(test_kor)), ncol=20)

n_fold <- 4
st.time <- proc.time()
for(i in 1:dim(test_kor)[1]) # 36) 
{ 
  ans.time <-proc.time()
  # ############## Test NA 탐색
  # i<-5;n<-2 ; i <-11 ; n<-2 ; 특별사례 i<-12 i<-9 ; n<-1 ; i<-1
  
  test_obs <- test_kor[i,]
  na.list <- which(is.na(test_obs))
  
  # 테스트 셋에서의 비-핵심 변수 제거
  test_obs <- test_obs[names(test_obs) %nin% del_list1]
  na.list1 <- which(is.na(test_obs))
  
  # ###########################################################################
  # ###########################################################################
  # ###########################################################################

  for(n in 1:length(na.list1)  )
  {
    ## 중요 변수 분석 시작
    
    start.time <- proc.time()
    
      result_mat[last.num+n,1]<- i+1
       # ###########################################################################
      result_mat[last.num+n,2]<- LETTERS[col_list %in% colnames(test_kor1)[is.na(test_kor1[i,])][n]][1]
      
       # 특별케이스의 사례 추출
      var_list <- test_obs[,!is.na(test_obs)]
      d_list <- dist_list[!is.na(test_obs)]
      dst <- c()
      for(dst.idx in 1:length(var_list))
        {
        if( (var_list[[dst.idx]]) %in% d_list[[dst.idx]] )
          {
          dst <- cbind(dst, as.matrix(var_list[dst.idx]) )
          #print( as.vector(var_list[dst.idx]))
          }
        }
    
    # ###########################################################################
    # ###########################################################################
    # 특별 사례
    
    if(length(dst) !=  0 )
    { 
      # 중요 변수 추출
      main_dist_dat <- dist_dat[names(test_obs) %nin% del_list1]
      # 타겟 변수 설정
      dist_y <- dist_dat[ ,na.list1[n]]
      # 결측 데이터 제거
      main_dist_dat1 <- main_dist_dat[,-na.list1]
     
      
      for(f in 3)#1:n_fold ) # f<-1
      {

        # ###########################################################################
        # ###########################################################################
        #  CAT
        if(class(dist_dat[,na.list1[n]])=="factor")
        {
          print("FACTOR_SPEC")
          set.seed(1234*f)
          # Test 처리
          
          
          # ID number 생성
          main_dist_dat_case <- cbind(main_dist_dat1,id=c(1:dim(main_dist_dat1)[1]))
          id <- c()
          for(case.idx in 1:length(dst))
          { 
            id.x<-main_dist_dat_case[main_dist_dat_case[,colnames(main_dist_dat_case) == colnames(dst)[case.idx]] %in% (dst)[1,case.idx] ,]$id
            id <- c(id,id.x)
          }
          
          # 특별케이스 추출 + Test obs 병합
          main_dist_dat_case1 <- rbind(main_dist_dat1[unique(sort(id)),],var_list)
          main_dist_dat_case1 <- main_dist_dat_case1[, colnames(main_dist_dat_case1) %nin% colnames(dst)]
          main_dist_dat_asc <- main_dist_dat_case1
          dist_y_case <- dist_y[unique(sort(id))]
          
          
          # 로그화 
          num_var <- main_dist_dat_case1[colnames(main_dist_dat_case1) %in% num_list1]
          main_dist_dat_case1[colnames(main_dist_dat_case1) %in% num_list1] <- log1p(main_dist_dat_case1[colnames(main_dist_dat_case1) %in% num_list1] )
          
          # 더미화
          main_dist_dum_dat1 <- fastDummies::dummy_cols(main_dist_dat_case1, 
                                                        select_columns=colnames(main_dist_dat_case1)[colnames(main_dist_dat_case1) %nin% num_list1],remove_first_dummy = FALSE)
          main_dist_dum_dat1 <- main_dist_dum_dat1[, colnames(main_dist_dum_dat1) %nin% cat_list1 ]
          
          
          # train의  타겟변수 수치화 
          dist_y_case<- as.factor(as.vector(dist_y_case))
          dist_num_y <- as.vector(dist_y_case)
          uni <- unique(dist_num_y)
          for(y.idx in 1:length(unique(dist_y_case)))
            {
              dist_num_y[dist_y_case==uni[y.idx]] <- y.idx
            }
          dist_num_y<- as.numeric(dist_num_y)
          
          # RF - 전체 데이터 
          tryCatch(randF<-randomForest::randomForest(y=as.factor(dist_y_case), x=main_dist_dat_case1[-dim(main_dist_dat_case1)[1],] ,importance=F,ntree=300),error=function(err) FALSE )
          tryCatch(p.test1<-predict(randF,newdata=main_dist_dat_case1[dim(main_dist_dat_case1)[1],]) ,error=function(err) FALSE )
          
          result_mat[last.num+n,3]<- as.character(p.test1)
          result_mat[last.num+n,11]<- 0.8
          
          #gau.time <- proc.time()
          # gaussain 유사도 
          ga_dist <- gausskernel(as.matrix(main_dist_dum_dat1),sigma=1)
          ga_dist1 <-  sort(ga_dist[dim(ga_dist)[1], -dim(ga_dist)[2]],decreasing = T)
          #table(dist_y_case[as.numeric(names(ga_dist1[1:ceiling(length(ga_dist1)*0.025)]))])
          result_mat[last.num+n,4]<- as.character(names(which.max(table(dist_y_case[as.numeric(names(ga_dist1[1:ceiling(length(ga_dist1)*0.025)]))]))))
          result_mat[last.num+n,12]<- 0.5
          #print( (proc.time() - gau.time)[3] )
          
          
          #dis.time <- proc.time()
          # Euclidean 거리 
          eucd_dist <- philentropy::distance(main_dist_dum_dat1, method = "euclidean")
          colnames(eucd_dist) <- 1:dim(eucd_dist)[2]
          eucd_dist1 <- sort(eucd_dist[dim(eucd_dist)[1], -dim(eucd_dist)[2]],decreasing = F)
          #table(dist_y_case[as.numeric(names(eucd_dist1[1:ceiling(length(eucd_dist1)*0.025)]))])
          
          result_mat[last.num+n,5]<- as.character(names(which.max(table(dist_y_case[as.numeric(names(eucd_dist1[1:ceiling(length(eucd_dist1)*0.025)]))]))))
          result_mat[last.num+n,13]<- 0.5
          
          
          # jensen_difference 거리 
          JD_dist <- philentropy::distance(main_dist_dum_dat1, method = "jensen_difference")
          colnames(JD_dist) <- 1:dim(JD_dist)[2]
          JD_dist1 <- sort(JD_dist[dim(JD_dist)[1], -dim(JD_dist)[2]],decreasing = F)
          #table(dist_y_case[as.numeric(names(JD_dist1[1:ceiling(length(JD_dist1)*0.025)]))])
          result_mat[last.num+n,6]<- as.character(names(which.max(table(dist_y_case[as.numeric(names(JD_dist1[1:ceiling(length(JD_dist1)*0.025)]))]))))
          result_mat[last.num+n,14]<- 0.5
          
          # Jaccard 거리 
          jacc_dist <- philentropy::distance(main_dist_dum_dat1, method = "jaccard")
          colnames(jacc_dist) <- 1:dim(jacc_dist)[2]
          jacc_dist1 <- sort(jacc_dist[dim(jacc_dist)[1], -dim(jacc_dist)[2]],decreasing = F)
          #table(dist_y_case[as.numeric(names(jacc_dist1[1:ceiling(length(jacc_dist1)*0.025)]))])
          result_mat[last.num+n,7]<- as.character(names(which.max(table(dist_y_case[as.numeric(names(jacc_dist1[1:ceiling(length(jacc_dist1)*0.025)]))]))))
          result_mat[last.num+n,15]<- 0.5
          #print( (proc.time() - dis.time)[3] )
          
          # association rule data 구축
          #ass.time <- proc.time()
          for(col.idx in 1:dim(main_dist_dat_asc)[2]  )
            {
            if(class(main_dist_dat_asc[,col.idx]) != "factor"   )
              {
                main_dist_dat_asc[main_dist_dat_asc[,col.idx] >1, col.idx] <- 2  
                main_dist_dat_asc[,col.idx] <- as.factor(main_dist_dat_asc[,col.idx])
              }
            }
          
          # make list
          main_dist_dat_asc1 <- cbind(main_dist_dat_asc[-dim(main_dist_dat_asc)[1],],target=as.factor(main_dist_dat[rownames(main_dist_dat) %in% as.numeric(rownames(main_dist_dat_asc)[-dim(main_dist_dat_asc)[1]]), na.list1[n]]))
          trans = as(main_dist_dat_asc1, "transactions")
          # make target
          aru_y <- as(main_dist_dat_asc[dim(main_dist_dat_asc)[1],] ,"transactions")
          # arules classifier
          
          tryCatch(classifier <- CBA(target ~ ., trans, supp = 0.05, conf=0.5),error=function(err) FALSE )
          tryCatch(pred <- predict(classifier, aru_y),error=function(err) FALSE )
          #rules(classifier)
          #inspect(rules(classifier))
          tryCatch(result_mat[last.num+n,8]<- as.character(pred),error=function(err) FALSE )
          tryCatch(result_mat[last.num+n,16]<- 0.7,error=function(err) FALSE )
          
          #print( (proc.time() - ass.time)[3] )
          
        }
        
        # ###########################################################################
        # ###########################################################################
        # NUM
        if(class(dist_dat[,na.list1[n]])!="factor")
        {
          print("NUM_SPEC")
          set.seed(1234*f)
          # Test 처리
          
          # ID number 생성
          main_dist_dat_case <- cbind(main_dist_dat1,id=c(1:dim(main_dist_dat1)[1]))
          id <- c()
          
          for(case.idx in 1:length(dst))
          { 
            id.x<-main_dist_dat_case[main_dist_dat_case[,colnames(main_dist_dat_case) == colnames(dst)[case.idx]] %in% (dst)[1,case.idx] ,]$id
            id <- c(id,id.x)
          }
          # 해당 특별케이스 추출 + Test obs 병합
          main_dist_dat_case1 <- rbind(main_dist_dat1[unique(sort(id)),],var_list)
          main_dist_dat_case1 <- main_dist_dat_case1[, colnames(main_dist_dat_case1) %nin% colnames(dst)]
          main_dist_dat_asc <- main_dist_dat_case1
          dist_y_case <- dist_y[unique(sort(id))]
          
          
          # 로그화
          num_var <- main_dist_dat_case1[colnames(main_dist_dat_case1) %in% num_list1]
          main_dist_dat_case1[colnames(main_dist_dat_case1) %in% num_list1] <- log1p(main_dist_dat_case1[colnames(main_dist_dat_case1) %in% num_list1] )
          
          main_dist_dum_dat1 <- fastDummies::dummy_cols(main_dist_dat_case1, 
                                                        select_columns=colnames(main_dist_dat_case1)[colnames(main_dist_dat_case1) %nin% num_list1],remove_first_dummy = FALSE)
          main_dist_dum_dat1 <- main_dist_dum_dat1[, colnames(main_dist_dum_dat1) %nin% cat_list1 ]
          

          
          # RF - 전체 데이터 
          ################################ RF_cat
          rf.time <- proc.time()
          
          dist_y_case_cat <- dist_y_case
          dist_y_case_cat[dist_y_case > 1] <- 2
          dist_y_case_cat[dist_y_case == 1] <- 1
          dist_y_case_cat[dist_y_case == 0] <- 0
          
          
          train_rf <- main_dist_dat_case1[-dim(main_dist_dat_case1)[1],]
          y_rf <- main_dist_dat_case1[dim(main_dist_dat_case1)[1],]
          
          tryCatch(randF<-randomForest::randomForest(y=as.factor(dist_y_case_cat), x=train_rf ,importance=F,ntree=300),error=function(err) FALSE )
              p.test_ft <-predict(randF,newdata=train_rf)
              p.test <-predict(randF,newdata=y_rf)
              y_cat01<-dist_y_case[p.test_ft %in% c(0,1)] 
              y_cat02<-dist_y_case[p.test_ft %in% c(2)]
              
              
              cat01<-train_rf[p.test_ft %in% c(0,1),] 
              cat02<-train_rf[p.test_ft %in% c(2),] 
              
              if(length(y_cat02) >= 2)
                {
                  randF_cat1 <-randomForest::randomForest(y=y_cat01, x=cat01,importance=F,ntree=300)
                  randF_cat2 <-randomForest::randomForest(y=y_cat02, x=cat02,importance=F,ntree=300)
                
                  ifelse(p.test %in% c(0,1),
                    final.test <- predict(randF_cat1,newdata=y_rf),
                    final.test <- predict(randF_cat2,newdata=y_rf))
                }
              
              if(length(y_cat02) < 2)
                {
                
                  randF_cat1 <-randomForest::randomForest(y=y_cat01, x=cat01,importance=F,ntree=300)
                  final.test <- predict(randF_cat1,newdata=y_rf)
                }
              result_mat[last.num+n,3]<- final.test
              result_mat[last.num+n,11]<- 1.25
              
              print( (proc.time() - rf.time)[3] )
              
              gau.time <- proc.time()
              # gaussain 유사도 
              ga_dist <- gausskernel(as.matrix(main_dist_dum_dat1),sigma=1)
              ga_dist1 <-  sort(ga_dist[dim(ga_dist)[1], -dim(ga_dist)[2]],decreasing = T)
              result_mat[last.num+n,4]<- round(mean(dist_y_case[as.numeric(names(ga_dist1[1:ceiling(length(ga_dist1)*0.025)]))]),0)
              result_mat[last.num+n,12]<- 0.5
              print( (proc.time() - gau.time)[3] )
              
              
              dis.time <- proc.time()
              # Euclidean 거리 
              eucd_dist <- philentropy::distance(main_dist_dum_dat1, method = "euclidean")
              colnames(eucd_dist) <- 1:dim(eucd_dist)[2]
              eucd_dist1 <- sort(eucd_dist[dim(eucd_dist)[1], -dim(eucd_dist)[2]],decreasing = F)
              result_mat[last.num+n,5]<- round(mean(dist_y_case[as.numeric(names(eucd_dist1[1:ceiling(length(eucd_dist1)*0.025)]))]),0)
              result_mat[last.num+n,13]<- 0.5
              
              
              # jensen_difference 거리 
              JD_dist <- philentropy::distance(main_dist_dum_dat1, method = "jensen_difference")
              colnames(JD_dist) <- 1:dim(JD_dist)[2]
              JD_dist1 <- sort(JD_dist[dim(JD_dist)[1], -dim(JD_dist)[2]],decreasing = F)
              result_mat[last.num+n,6]<- round(mean(dist_y_case[as.numeric(names(JD_dist1[1:ceiling(length(JD_dist1)*0.025)]))]),0)
              result_mat[last.num+n,14]<- 0.5
              

              # Jaccard 거리 
              jacc_dist <- philentropy::distance(main_dist_dum_dat1, method = "jaccard")
              colnames(jacc_dist) <- 1:dim(jacc_dist)[2]
              jacc_dist1 <- sort(jacc_dist[dim(jacc_dist)[1], -dim(jacc_dist)[2]],decreasing = F)
              result_mat[last.num+n,7]<- round(mean(dist_y_case[as.numeric(names(jacc_dist1[1:ceiling(length(jacc_dist1)*0.025)]))]),0)
              result_mat[last.num+n,15]<- 0.5
              print( (proc.time() - dis.time)[3] )
              
        }
        tot.time <- proc.time() - start.time
        print(paste0("TOT_",tot.time))
      }
        
      }
    
    
    # ###########################################################################
    # ###########################################################################
    # 일반 사례
    if( length(dst) ==  0 )
    { 
      
      main_pred_dat <- pred_dat[names(test_obs) %nin% del_list1]
      
      pred_y <- pred_dat[ ,na.list1[n]]
      main_pred_dat1 <- main_pred_dat[,-na.list1]
      
      for(f in 3)#1:n_fold )
      {
        start.time <- proc.time()
        
        # ###########################################################################
        # Start the cat Analysis  
        
        if(class(pred_dat[,na.list1[n]])=="factor")
        {      
          ### n<- 2 ; i<-1
          # Predict_model
          print("FACTOR_NOMAL")
          set.seed(1234*f)
          val.idx <- split(sample(1:dim(main_pred_dat1)[1]),1:n_fold)
          
          # NB용 데이터- 로그화x
          pred_val_dat_nb <- main_pred_dat1[val.idx[[f]],]
          pred_X_dat1_nb <- main_pred_dat1[-val.idx[[f]],]
          
          # 로그화 
          num_var <- main_pred_dat1[colnames(main_pred_dat1) %in% num_list1]
          main_pred_dat1 <- rbind(main_pred_dat1,var_list)
          main_pred_dat1[colnames(main_pred_dat1) %in% num_list1] <- log1p(main_pred_dat1[colnames(main_pred_dat1) %in% num_list1] )
          
          # 더미화
          main_pred_dum_dat1 <- fastDummies::dummy_cols(main_pred_dat1, 
                                                        select_columns=colnames(main_pred_dat1)[colnames(main_pred_dat1) %nin% num_list1],remove_first_dummy = FALSE)
          main_pred_dum_dat1 <- main_pred_dum_dat1[, colnames(main_pred_dum_dat1) %nin% cat_list1 ]
          
          main_pred_dum_dat2 <- fastDummies::dummy_cols(main_pred_dat1, 
                                                        select_columns=colnames(main_pred_dat1)[colnames(main_pred_dat1) %nin% num_list1],remove_first_dummy = TRUE)
          main_pred_dum_dat2 <- main_pred_dum_dat2[, colnames(main_pred_dum_dat2) %nin% cat_list1 ]
          
          # test obs
          main_test <-  main_pred_dat1[dim(main_pred_dat1)[1],]
          main_dum_test1 <-main_pred_dum_dat1[dim(main_pred_dum_dat1)[1],]
          main_dum_test2 <-main_pred_dum_dat2[dim(main_pred_dum_dat2)[1],]
          
          # train set
          main_pred_dat1<-main_pred_dat1[-dim(main_pred_dat1)[1],]
          main_pred_dum_dat1<-main_pred_dum_dat1[-dim(main_pred_dum_dat1)[1],]
          main_pred_dum_dat2<-main_pred_dum_dat2[-dim(main_pred_dum_dat2)[1],]
          
          # set the validation
          
          pred_val_dat1 <- main_pred_dat1[val.idx[[f]],]
          pred_val_dum_dat1 <- main_pred_dum_dat1[val.idx[[f]],]
          pred_val_dum_dat2 <- main_pred_dum_dat2[val.idx[[f]],]
          pred_X_dat1 <- main_pred_dat1[-val.idx[[f]],]
          pred_X_dum_dat1 <- main_pred_dum_dat1[-val.idx[[f]],]
          pred_X_dum_dat2 <- main_pred_dum_dat2[-val.idx[[f]],]
          
          pred_val_y <- as.factor(as.vector(pred_y[val.idx[[f]]]))
          pred_y_y <- as.factor(as.vector(pred_y[-val.idx[[f]]]))
          
          uni <- unique(pred_val_y)
          pred_val_num_y <- as.vector(pred_val_y)
          pred_y_num_y <- as.vector(pred_y_y)
          for(y.idx in 1:length(unique(pred_val_y)))
            {
              pred_val_num_y[pred_val_num_y==uni[y.idx]] <- y.idx
              pred_y_num_y[pred_y_num_y==uni[y.idx]] <- y.idx
            }
          pred_val_num_y<- as.numeric(pred_val_num_y)
          pred_y_num_y<- as.numeric(pred_y_num_y)
          
          
          ls.time <- proc.time()
          ####################### LASSO
          cv.glm.fit <- cv.glmnet(x= as.matrix(pred_X_dum_dat2),y=pred_y_num_y,family="multinomial",alpha=1,type.measure="class",nfold=3) 
          
          coef_mat <- matrix(0,nrow=dim(pred_X_dum_dat2)[2]+1 , ncol=length(unique(pred_y_num_y)))
          for(c in 1:length(coef(cv.glm.fit)))
            {
              coef_mat[,c]<-coef(cv.glm.fit, s = "lambda.1se")[[c]][,1]
            }
          f.mat <- exp(cbind(1,as.matrix(pred_val_dum_dat2))%*% coef_mat) 
          y.table<-table(pred_val_num_y,(apply(f.mat/rowSums(f.mat),1,which.max)))
          
          f.mat <- exp(cbind(1,as.matrix(main_dum_test2))%*% coef_mat) 
          result_mat[last.num+n,3]<- as.character(uni[apply(f.mat/rowSums(f.mat),1,which.max)])
          result_mat[last.num+n,11]<- sum(diag(y.table[rownames(y.table) %in% colnames(y.table),colnames(y.table) %in% rownames(y.table) ]))/sum(y.table)
          
          print( (proc.time() - ls.time)[3] )
          
          ent.time <- proc.time()
          ####################### Enet
          cv.glm.fit <- cv.glmnet(x= as.matrix(pred_X_dum_dat2),y=pred_y_num_y,family="multinomial",alpha=0.8,type.measure="class",nfold=3) 
          coef_mat <- matrix(0,nrow=dim(pred_X_dum_dat2)[2]+1 , ncol=length(unique(pred_y_num_y)))
          for(c in 1:length(coef(cv.glm.fit)))
            {
              coef_mat[,c]<-coef(cv.glm.fit, s = "lambda.1se")[[c]][,1]
            }
          f.mat <- exp(cbind(1,as.matrix(pred_val_dum_dat2))%*% coef_mat) 
          y.table<-table(pred_val_num_y,(apply(f.mat/rowSums(f.mat),1,which.max)))
          
          f.mat <- exp(cbind(1,as.matrix(main_dum_test2))%*% coef_mat) 
          result_mat[last.num+n,4]<- as.character(uni[apply(f.mat/rowSums(f.mat),1,which.max)])
          result_mat[last.num+n,12]<- sum(diag(y.table[rownames(y.table) %in% colnames(y.table),colnames(y.table) %in% rownames(y.table) ]))/sum(y.table)
          
          print( (proc.time() - ent.time)[3] )
          
          nb.time <- proc.time()
          ####################### NaiveBayes ## factor
          pred_X_dat_cat1 <- pred_X_dat1_nb
          pred_val_dat_cat1 <- rbind(pred_val_dat_nb,var_list)
          
          for(ct.idx in 1:length(pred_X_dat1[,colnames(pred_X_dat1) %in% num_list1])  )
            {
            pred_X_dat_cat1[ , colnames(pred_X_dat_cat1)[colnames(pred_X_dat_cat1) %in% num_list1][ct.idx]] <- ifelse(
              pred_X_dat_cat1[ , colnames(pred_X_dat_cat1)[colnames(pred_X_dat_cat1) %in% num_list1][ct.idx]] >1 ,  
                      as.factor(2), ifelse( pred_X_dat_cat1[ , colnames(pred_X_dat_cat1)[colnames(pred_X_dat_cat1) %in% num_list1][ct.idx]] == 1,as.factor(1),as.factor(0)))
            
            pred_val_dat_cat1[ , colnames(pred_val_dat_cat1)[colnames(pred_val_dat_cat1) %in% num_list1][ct.idx]] <- ifelse(
              pred_val_dat_cat1[ , colnames(pred_val_dat_cat1)[colnames(pred_val_dat_cat1) %in% num_list1][ct.idx]] >1 ,  as.factor(2),
              ifelse( pred_val_dat_cat1[ , colnames(pred_val_dat_cat1)[colnames(pred_val_dat_cat1) %in% num_list1][ct.idx]] == 1,as.factor(1),as.factor(0)))
          }
          pred_val_dat_cat1 <- pred_val_dat_cat1[-dim(pred_val_dat_cat1)[1],]
          nb.fit<-e1071::naiveBayes(pred_X_dat_cat1,pred_y_y)
          y.table<-table( pred_val_y ,predict(nb.fit,pred_val_dat_cat1,type = "class"))
          
          result_mat[last.num+n,5]<- as.character(predict(nb.fit,pred_val_dat_cat1[dim(pred_val_dat_cat1)[1],],type = "class"))
          result_mat[last.num+n,13]<- sum(diag(y.table[rownames(y.table) %in% colnames(y.table),colnames(y.table) %in% rownames(y.table) ]))/sum(y.table)
          
          print( (proc.time() - nb.time)[3] )
          
          rf.time <- proc.time()
          ####################### random Forest
          
          randF<-randomForest::randomForest(y=as.factor(pred_y_y), x=pred_X_dat1,importance=F,ntree=300)
          p.test1<-predict(randF,newdata=pred_val_dat1 )
          y.table<-table(pred_val_y,p.test1)
          
          result_mat[last.num+n,6]<- as.character(predict(randF,newdata=main_test ))
          result_mat[last.num+n,14]<- sum(diag(y.table[rownames(y.table) %in% colnames(y.table),colnames(y.table) %in% rownames(y.table) ]))/sum(y.table)
          print( (proc.time() - rf.time)[3] )
          
          et.time <- proc.time()
          ####################### Extra Trees
          et <- extraTrees(x=as.matrix(pred_X_dum_dat2),y=pred_y_y, numRandomCuts = 2)
          yhat <- predict(et, as.matrix(pred_val_dum_dat2))
          
          y.table<-table(pred_val_y,yhat)
          result_mat[last.num+n,7]<- as.character(predict(et, as.matrix(main_dum_test2)) )
          result_mat[last.num+n,15]<- sum(diag(y.table[rownames(y.table) %in% colnames(y.table),colnames(y.table) %in% rownames(y.table) ]))/sum(y.table)
          print( (proc.time() - et.time)[3] )
          
          svm.time <- proc.time()
          ####################### svm - factor
          svm.model<-e1071::svm(x=as.matrix(pred_X_dum_dat2),y=pred_y_y,kernel="radial") #,gamma=gamma,cost=cost)
          pred.svm<-predict(svm.model,as.matrix(pred_val_dum_dat2))
          #print(caret::confusionMatrix(pred_val_y,pred.svm)$overall[[1]])
          y.table<-table(pred_val_y,pred.svm)
          
          result_mat[last.num+n,8]<- as.character(predict(svm.model,as.matrix(main_dum_test2)) )
          result_mat[last.num+n,16]<- sum(diag(y.table[rownames(y.table) %in% colnames(y.table),colnames(y.table) %in% rownames(y.table) ]))/sum(y.table)
          
          print( (proc.time() - svm.time)[3] )
          
          knn.time <- proc.time()
          ####################### knn
          sim<-2
          cls<- c(1,2,3,seq(5,20,3))
          clust<-matrix(0,nrow =sim, ncol=length(cls) )
          colnames(clust) <- cls
          
          for(sim.idx in 1:sim)
          {
            id1<-split(sample(1:dim(pred_X_dum_dat1)[1]),1:4)
            knn.train_x.mat<-pred_X_dum_dat1[-id1[[1]],]
            knn.val_x.mat<-pred_X_dum_dat1[id1[[1]],]
            
            knn.train_y.vec<-pred_y_y[-id1[[1]]]
            knn.val_y.vec<-pred_y_y[id1[[1]]]
            
            
            
            for(cls.idx in 1:length(cls)){
              c<-knn.train_y.vec
              knn<-class::knn(knn.train_x.mat,knn.val_x.mat, cl=c,k=cls[cls.idx],prob = F)
              clust[sim.idx,cls.idx]<-mean(knn==knn.val_y.vec)
              
            }
          }
          # best.clust<-which.max(colMeans(clust))
          best.clust<-which.max(colMeans(clust))
          final.knn<-class::knn(train = pred_X_dum_dat1, test=pred_val_dum_dat1 , cl=pred_y_y,k=as.numeric(names(best.clust)))
          y.table<-table(pred_val_y,final.knn)
          
          result_mat[last.num+n,9]<- as.character( class::knn(train = pred_X_dum_dat1, test=main_dum_test1 , cl=pred_y_y,k=as.numeric(names(best.clust))) )
          result_mat[last.num+n,17]<- sum(diag(y.table[rownames(y.table) %in% colnames(y.table),colnames(y.table) %in% rownames(y.table) ]))/sum(y.table)
          
          #print(caret::confusionMatrix(final.knn , pred_val_y)$overall[[1]])
          print( (proc.time() - knn.time)[3] )
        }
        
        # ###########################################################################          
        # ###########################################################################
        
        # Start the NUM Analysis  
        if( class(pred_dat[,na.list1[n]]) != "factor")
        {
          print("NUM_NOMAL")
          # 로그화
          num_var <- main_pred_dat1[colnames(main_pred_dat1) %in% num_list1]
          main_pred_dat1 <- rbind(main_pred_dat1,var_list)
          main_pred_dat1[colnames(main_pred_dat1) %in% num_list1] <- log1p(main_pred_dat1[colnames(main_pred_dat1) %in% num_list1] )
          
          main_pred_dum_dat1 <- fastDummies::dummy_cols(main_pred_dat1, 
                                                        select_columns=colnames(main_pred_dat1)[colnames(main_pred_dat1) %nin% num_list1],remove_first_dummy = TRUE)
          main_pred_dum_dat1 <- main_pred_dum_dat1[, colnames(main_pred_dum_dat1) %nin% cat_list1 ]
          
          # test obs
          main_test <-  main_pred_dat1[dim(main_pred_dat1)[1],]
          main_dum_test1 <-main_pred_dum_dat1[dim(main_pred_dum_dat1)[1],]
          
          
          # train set
          main_pred_dat1<-main_pred_dat1[-dim(main_pred_dat1)[1],]
          main_pred_dum_dat1<-main_pred_dum_dat1[-dim(main_pred_dum_dat1)[1],]

          # set the validation
          val.idx <- split(sample(1:dim(main_pred_dat1)[1]),1:4)
          
          pred_val_dat1 <- main_pred_dat1[val.idx[[f]],]
          pred_val_dum_dat1 <- main_pred_dum_dat1[val.idx[[f]],]
          
          pred_X_dat1 <- main_pred_dat1[-val.idx[[f]],]
          pred_X_dum_dat1 <- main_pred_dum_dat1[-val.idx[[f]],]
          
          pred_val_y <- as.vector(pred_y[val.idx[[f]]])
          pred_y_y <- as.vector(pred_y[-val.idx[[f]]])
          
          pred_val_y_cat <- pred_val_y
          pred_val_y_cat[pred_val_y > 1] <- 2
          pred_val_y_cat[pred_val_y == 1] <- 1
          pred_val_y_cat[pred_val_y == 0] <- 0
          
          pred_y_y_cat <- pred_y_y
          pred_y_y_cat[pred_y_y > 1] <- 2
          pred_y_y_cat[pred_y_y == 1] <- 1
          pred_y_y_cat[pred_y_y == 0] <- 0
          

          ################################ RF_cat
          randF<-randomForest::randomForest(y=as.factor(pred_y_y_cat), x=pred_X_dat1,ntree=300,importance=F)
          p.test_ft<-predict(randF,newdata=pred_X_dat1 )
          p.val_ft<-predict(randF,newdata=pred_val_dat1 )
          p.test <- predict(randF,newdata=main_test  )
          #table(p.val_ft,pred_val_y_cat)
          cat01<-pred_X_dat1[p.test_ft %in% c(0,1),] 
          cat02<-pred_X_dat1[p.test_ft %in% c(2),]
          
          cat01_val<-pred_val_dat1[p.val_ft %in% c(0,1),] 
          cat02_val<-pred_val_dat1[p.val_ft %in% c(2),] 
          
          y_cat01<-pred_y_y[p.test_ft %in% c(0,1)] 
          y_cat02<-pred_y_y[p.test_ft %in% c(2)]
          
          if(length(y_cat02) >= 2){
            
            randF_cat1<-randomForest::randomForest(y=y_cat01, x=cat01,importance=F,ntree=300,type="regression")
            p.test_sc1<-predict(randF_cat1,newdata=cat01_val )
            
            randF_cat2<-randomForest::randomForest(y=y_cat02, x=cat02,importance=F,ntree=300,type="regression")
            p.test_sc2<-predict(randF_cat2,newdata=cat02_val )
            
             ifelse(p.test %in% c(0,1),
                   final.test <- predict(randF_cat1,newdata=main_test),
                   final.test <- predict(randF_cat2,newdata=main_test))

             acc <- sum(( c(p.test_sc1,p.test_sc2) - c(pred_val_y[p.val_ft %in% c(0,1)],pred_val_y[p.val_ft %in% c(2)]))^2)/length(pred_val_y)
            result_mat[last.num+n,3]<- as.character( final.test)
            result_mat[last.num+n,11]<- 1/acc
            
            
            # gaussain 유사도 
            gau.time <- proc.time()
            cat_dum_01<-rbind(cat01,main_test)
            cat_dum_01 <- fastDummies::dummy_cols(cat_dum_01, 
                                                  select_columns=colnames(cat_dum_01)[colnames(cat_dum_01) %nin% num_list1],remove_first_dummy = FALSE)
            cat_dum_01 <- cat_dum_01[, colnames(cat_dum_01) %nin% cat_list1 ]
            cat_dum_02<-rbind(cat02,main_test)
            cat_dum_02 <- fastDummies::dummy_cols(cat_dum_02, 
                                                  select_columns=colnames(cat_dum_02)[colnames(cat_dum_02) %nin% num_list1],remove_first_dummy = FALSE)
            cat_dum_02 <- cat_dum_02[, colnames(cat_dum_02) %nin% cat_list1 ]
            
            ifelse(p.test %in% c(0,1),ga_dist <- gausskernel(as.matrix(cat_dum_01),sigma=1),
                   ga_dist <- gausskernel(as.matrix(cat_dum_02),sigma=1) )
            ga_dist1 <-  sort(ga_dist[dim(ga_dist)[1], -dim(ga_dist)[2]],decreasing = T)
            
            ifelse(p.test %in% c(0,1),result_mat[last.num+n,4]<-  round(mean(y_cat01[as.numeric(names(ga_dist1[1:ceiling(length(ga_dist1)*0.001)]))]),0),
                   round(result_mat[last.num+n,4]<-  mean(y_cat02[as.numeric(names(ga_dist1[1:ceiling(length(ga_dist1)*0.001)]))]),0) )
            result_mat[last.num+n,12]<- 0.5 
            print( (proc.time() - gau.time)[3] )
            
          }

          if(length(y_cat02) < 2){
            randF<-randomForest::randomForest(y=y_cat01, x=cat01,importance=F,ntree=300)
            randF_cat1<-predict(randF,newdata=cat01 )
            final.test <- predict(randF_cat1,newdata=main_test)
            acc <-  (sum(( c(p.test_sc1) - c(pred_val_y[p.test_ft %in% c(0,1)]))^2)/length(pred_val_y))
            
            result_mat[last.num+n,3]<- as.character( final.test)
            result_mat[last.num+n,11]<- 1/acc

            # gaussain 유사도 
            gau.time <- proc.time()
            cat_dum_01<-rbind(cat01,main_test)
            cat_dum_01 <- fastDummies::dummy_cols(cat_dum_01, 
                                                          select_columns=colnames(cat_dum_01)[colnames(cat_dum_01) %nin% num_list1],remove_first_dummy = FALSE)
            cat_dum_01 <- cat_dum_01[, colnames(cat_dum_01) %nin% cat_list1 ]
            
            ga_dist <- gausskernel(as.matrix(cat_dum_01),sigma=1)
            ga_dist1 <-  sort(ga_dist[dim(ga_dist)[1], -dim(ga_dist)[2]],decreasing = T)
            print(length(ga_dist1))
            result_mat[last.num+n,4]<-  round(mean(y_cat01[as.numeric(names(ga_dist1[1:ceiling(length(ga_dist1)*0.001)]))]),0)
            result_mat[last.num+n,12]<- 0.5
            print( (proc.time() - gau.time)[3] )
          
          }
          
          cv.glm.fit <- cv.glmnet(x= as.matrix(pred_X_dum_dat1),y= pred_y_y ,family="gaussian",
                                  alpha=0.8,type.measure="deviance",nfold=3)
          f.mat <- cbind(1,as.matrix(pred_val_dum_dat1))%*% coef(cv.glm.fit, s = "lambda.1se")
          print(sum(( (f.mat) - pred_val_y)^2)/length(pred_val_y))
          result_mat[last.num+n,5]<- c(1,as.matrix(main_dum_test1))%*% coef(cv.glm.fit, s = "lambda.1se")[,1]
          result_mat[last.num+n,13]<- 1/(sum(( (f.mat) - pred_val_y)^2)/length(pred_val_y))
          
        }
        

      }
      }
      tot.time <-(proc.time() - start.time)[3]
      print(paste0("TOT_",tot.time))
      
    }
  
  
  # ###########################################################################
  # ###########################################################################
  # ###########################################################################
  # Start Non-important Variable
  NIV.time <- proc.time()
  
  if(sum(names(test_kor[i,][na.list]) %in% del_list1)>0)
  {
    # 테스트 셋 서치
    result_obs<- test_kor[i,]
    # 주요 변수 
    data_obs<- test_kor[i, colnames(result_obs) %nin% del_list1]
    
    # 비 주요 변수 - 타겟
    non_im_var <- del_list1[del_list1  %in%  colnames(result_obs[na.list])]
    
    for(nim.idx in 1:length(non_im_var))
      {
      
      # ###########################################################################
      # ###########################################################################
      # 사상자수
      
        if(non_im_var[nim.idx] =="사상자수")
          {

          # 채우기
          for(n.idx in 1:sum(is.na(data_obs)))
            {
            
            # 수치 결과 채우기
            if(colnames(data_obs)[is.na(data_obs)][n.idx] %in% num_list1 )
              {
                result <- as.numeric(result_mat[last.num + n.idx , 3:10 * (result_mat[last.num+n,11:18] != 0) ])
                weight <- as.numeric(result_mat[last.num + n.idx , 11:20 * (result_mat[last.num+n,11:20] != 0) ])
                weight <- weight/sum(weight)
                
                result_obs[,match(result_mat[last.num + n.idx,2],LETTERS)] <- sum(result*weight)
              }
            
            # 범주 결과 채우기
            if(colnames(data_obs)[is.na(data_obs)][n.idx] %nin% num_list1 )
              {
              
              result <- as.character(result_mat[last.num + n.idx , 3:10 * (result_mat[last.num+n,11:18] != 0) ])
              weight <- as.numeric(result_mat[last.num + n.idx , 11:20 * (result_mat[last.num+n,11:20] != 0) ])
              weight <- weight/sum(weight)
              
              
              uni.rslt <- unique(result)
              wt.vote <- matrix(0, nrow =  1,ncol=length(uni.rslt))
              colnames(wt.vote) <- uni.rslt 
              
                for(uni.idx in 1:length(uni.rslt))
                {
                  wt.vote[1,uni.idx] <- sum(weight[result == uni.rslt[uni.idx]])
                
                }
              
              result_obs[,match(result_mat[last.num + n.idx,2],LETTERS)] <- colnames(wt.vote)[which.max(wt.vote)]
              }
          }
          
          # ###########################################################################
          # 사상자수 예측 

          result_mat[last.num+n+nim.idx,1]<- i + 1
          result_mat[last.num+n+nim.idx,2]<- LETTERS[which(colnames(result_obs) %in% "사상자수")][1]
          result_mat[last.num+n+nim.idx,3]<- sum(result_obs[, num_list1])
          result_obs[ , "사상자수"] <- sum(result_obs[, num_list1])

        }
      
      # ###########################################################################
      # ###########################################################################
      # 도로형태
      
      if(non_im_var[nim.idx] == "도로형태")
      {
        
        # 채우기
        for(n.idx in 1:sum(is.na(data_obs)))
        {
          
          # 수치 결과 채우기
          if(colnames(data_obs)[is.na(data_obs)][n.idx] %in% num_list2 )
          {
            result <- as.numeric(result_mat[last.num + n.idx , 3:10 * (result_mat[last.num+n,11:18] != 0) ])
            weight <- as.numeric(result_mat[last.num + n.idx , 11:20 * (result_mat[last.num+n,11:20] != 0) ])
            weight <- weight/sum(weight)
            
            # test에서 빈칸 찾기 -> 계산 된 값 채우기
            result_obs[,match(result_mat[last.num + n.idx,2],LETTERS)] <- sum(result*weight)
          }
          
          
          # 범주 결과 채우기
          if(colnames(data_obs)[is.na(data_obs)][n.idx] %nin% num_list2 )
          {
            
            result <- as.character(result_mat[last.num + n.idx , 3:10 * (result_mat[last.num+n,11:18] != 0) ])
            weight <- as.numeric(result_mat[last.num + n.idx , 11:20 * (result_mat[last.num+n,11:20] != 0) ])
            weight <- weight/sum(weight)
            
            
            uni.rslt <- unique(result)
            wt.vote <- matrix(0, nrow =  1,ncol=length(uni.rslt))
            colnames(wt.vote) <- uni.rslt 
            
            for(uni.idx in 1:length(uni.rslt))
            {
              wt.vote[1,uni.idx] <- sum(weight[result == uni.rslt[uni.idx]])
              
            }
            
            result_obs[,match(result_mat[last.num + n.idx,2],LETTERS)] <- colnames(wt.vote)[which.max(wt.vote)]
          }
          
        }
        
        # ###########################################################################
        # 도로형태 예측 
        result_mat[last.num+n+nim.idx,1]<- i + 1
        result_mat[last.num+n+nim.idx,2]<- LETTERS[which(colnames(result_obs) %in% "도로형태")][1]
        
        # 분석 데이터 구축
        final_list<-colnames(result_obs[,!is.na(result_obs)])
        
        final_data <- data_kor
        
        load<-unique(data_kor$도로형태_대분류)
        load_list<-list()
        for(load.idx in 1:length(load) ){
          
          load_list1 <- unique(data_kor[data_kor$발생지시도 %in% load[load.idx],colnames(data_kor) %in% "도로형태" ])
          load_list[load.idx] <- list(load_list1)
          
        }
        names(load_list) <- load

        dist_y_case<-final_data$도로형태
        final_data<-rbind(final_data[,colnames(final_data)%nin% "도로형태"],result_obs[,!is.na(result_obs)])
        final_data <- final_data[,colnames(final_data) %nin% "발생지시군구"]
        final_data1 <- final_data[final_data$도로형태_대분류 %in% result_obs$도로형태_대분류, colnames(final_data) %nin% "도로형태_대분류"]
        final_data1 <- final_data1[final_data1$사고유형_대분류 %in% result_obs$사고유형_대분류, colnames(final_data1) %nin% "사고유형_대분류"]
        #final_data1 <- final_data[final_data$사고유형_중분류 %in% result_obs$사고유형_중분류,]
        final_data1 <- final_data1[final_data1$발생지시도 %in% result_obs$발생지시도,colnames(final_data1) %nin% "발생지시도"]
        final_data1 <- final_data1[,colnames(final_data1) %nin% "도로형태"]
        
        num_var <- final_data1[colnames(final_data1) %in% num_list2]

        final_data1[colnames(final_data1) %in% num_list2] <- log1p(final_data1[colnames(final_data1) %in% num_list2] )
        
        final_dum_dat1 <- fastDummies::dummy_cols(final_data1, 
                                                  select_columns=colnames(final_data1)[colnames(final_data1) %nin% num_list1],remove_first_dummy = FALSE)
        final_dum_dat1 <- final_dum_dat1[, colnames(final_dum_dat1) %nin% cat_list1 ]
        
        gau.time <- proc.time()
        # gaussain 유사도 
        ga_dist <- gausskernel(as.matrix(final_dum_dat1),sigma=1)
        ga_dist1 <-  sort(ga_dist[dim(ga_dist)[1], -dim(ga_dist)[2]],decreasing = T)
        #table(dist_y_case[as.numeric(names(ga_dist1[1:ceiling(length(ga_dist1)*0.025)]))])
        result_mat[last.num+n+nim.idx,3]<- as.character(names(which.max(table(dist_y_case[as.numeric(names(ga_dist1[1:ceiling(length(ga_dist1)*0.025)]))]))))
        result_mat[last.num+n+nim.idx,11]<- 0.5
        print( (proc.time() - gau.time)[3] )

        
        dis.time <- proc.time()
        # Euclidean 거리 
        eucd_dist <- philentropy::distance(final_dum_dat1, method = "euclidean")
        colnames(eucd_dist) <- 1:dim(eucd_dist)[2]
        eucd_dist1 <- sort(eucd_dist[dim(eucd_dist)[1], -dim(eucd_dist)[2]],decreasing = F)
        #table(dist_y_case[as.numeric(names(eucd_dist1[1:ceiling(length(eucd_dist1)*0.025)]))])
        
        result_mat[last.num+n+nim.idx,4]<- as.character(names(which.max(table(dist_y_case[as.numeric(names(eucd_dist1[1:ceiling(length(eucd_dist1)*0.025)]))]))))
        result_mat[last.num+n+nim.idx,12]<- 0.5
        
        
        # jensen_difference 거리 
        JD_dist <- philentropy::distance(final_dum_dat1, method = "jensen_difference")
        colnames(JD_dist) <- 1:dim(JD_dist)[2]
        JD_dist1 <- sort(JD_dist[dim(JD_dist)[1], -dim(JD_dist)[2]],decreasing = F)
        #table(dist_y_case[as.numeric(names(JD_dist1[1:ceiling(length(JD_dist1)*0.025)]))])
        result_mat[last.num+n+nim.idx,5]<- as.character(names(which.max(table(dist_y_case[as.numeric(names(JD_dist1[1:ceiling(length(JD_dist1)*0.025)]))]))))
        result_mat[last.num+n+nim.idx,13]<- 0.5
        
        # Jaccard 거리 
        jacc_dist <- philentropy::distance(final_dum_dat1, method = "jaccard")
        colnames(jacc_dist) <- 1:dim(jacc_dist)[2]
        jacc_dist1 <- sort(jacc_dist[dim(jacc_dist)[1], -dim(jacc_dist)[2]],decreasing = F)
        #table(dist_y_case[as.numeric(names(jacc_dist1[1:ceiling(length(jacc_dist1)*0.025)]))])
        result_mat[last.num+n+nim.idx,6]<- as.character(names(which.max(table(dist_y_case[as.numeric(names(jacc_dist1[1:ceiling(length(jacc_dist1)*0.025)]))]))))
        result_mat[last.num+n+nim.idx,14]<- 0.5
        
      }
      
      
      
      # ###########################################################################
      # ###########################################################################
      # 발생지시군구 / 마지막 단계
      if(non_im_var[nim.idx] == "발생지시군구")
      {
        
        # 채우기
        for(n.idx in 1:sum(is.na(data_obs)))
        {
          
          # 수치 결과 채우기
          if(colnames(data_obs)[is.na(data_obs)][n.idx] %in% num_list1 )
          {
            result <- as.numeric(result_mat[last.num + n.idx , 3:10 * (result_mat[last.num+n,11:18] != 0) ])
            weight <- as.numeric(result_mat[last.num + n.idx , 11:20 * (result_mat[last.num+n,11:20] != 0) ])
            weight <- weight/sum(weight)
            
            # test에서 빈칸 찾기 -> 계산 된 값 채우기
            result_obs[,match(result_mat[last.num + n.idx,2],LETTERS)] <- sum(result*weight)
          }
          
          
          # 범주 결과 채우기
          if(colnames(data_obs)[is.na(data_obs)][n.idx] %nin% num_list1 )
          {
            
            result <- as.character(result_mat[last.num + n.idx , 3:10 * (result_mat[last.num+n,11:18] != 0) ])
            weight <- as.numeric(result_mat[last.num + n.idx , 11:20 * (result_mat[last.num+n,11:20] != 0) ])
            weight <- weight/sum(weight)
            
            
            uni.rslt <- unique(result)
            wt.vote <- matrix(0, nrow =  1,ncol=length(uni.rslt))
            colnames(wt.vote) <- uni.rslt 
            
            for(uni.idx in 1:length(uni.rslt))
            {
              wt.vote[1,uni.idx] <- sum(weight[result == uni.rslt[uni.idx]])
              
            }
            result_obs[,match(result_mat[last.num + n.idx,2],LETTERS)] <- colnames(wt.vote)[which.max(wt.vote)]
          }
          
        }
        
        
        # ###########################################################################
        # 발생지시군구 예측 
        result_mat[last.num+n+nim.idx,1]<- i + 1
        result_mat[last.num+n+nim.idx,2]<- LETTERS[which(colnames(result_obs) %in% "발생지시군구")][1]
        # 분석 데이터 구축
        final_list<-colnames(result_obs[,!is.na(result_obs)])
        
        final_data <- data_kor
        
        land<-unique(data_kor$발생지시도)
        land_list<-list()
        for(land.idx in 1:length(land) ){
          
          land_list1 <- unique(data_kor[data_kor$발생지시도 %in% land[land.idx],colnames(data_kor) %in% "발생지시군구" ])
          land_list[land.idx] <- list(land_list1)
          
        }
        names(land_list) <- land
        
        #land_list[names(land_list) %in% result_obs$발생지시도]
        
        dist_y_case<-final_data$발생지시군구
       
        final_data<-rbind(final_data[,colnames(final_data)%nin% "발생지시군구"],result_obs[,!is.na(result_obs)])
        
        final_data1 <- final_data[final_data$발생지시도 %in% result_obs$발생지시도,]
        final_data1 <- final_data1[final_data1$사고유형_대분류 %in% result_obs$사고유형_대분류, colnames(final_data1) %nin% "사고유형_대분류"]
        final_data1 <- final_data1[final_data1$도로형태_대분류 %in% result_obs$도로형태_대분류, colnames(final_data1) %nin% "도로형태_대분류"]
        final_data1 <- final_data1[final_data1$주야 %in% result_obs$주야, colnames(final_data1) %nin% "주야"]
        final_data1 <- final_data1[,colnames(final_data1) %nin% "발생지시도"]
        
        num_var <- final_data1[colnames(final_data1) %in% num_list1]
        final_data1[colnames(final_data1) %in% num_list1] <- log1p(final_data1[colnames(final_data1) %in% num_list1] )
        
        final_dum_dat1 <- fastDummies::dummy_cols(final_data1, 
                                                  select_columns=colnames(final_data1)[colnames(final_data1) %nin% num_list1],remove_first_dummy = FALSE)
        final_dum_dat1 <- final_dum_dat1[, colnames(final_dum_dat1) %nin% cat_list1 ]

        
        gau.time <- proc.time()
        # gaussain 유사도 
        ga_dist <- gausskernel(as.matrix(final_dum_dat1),sigma=1)
        ga_dist1 <-  sort(ga_dist[dim(ga_dist)[1], -dim(ga_dist)[2]],decreasing = T)
        #table(dist_y_case[as.numeric(names(ga_dist1[1:ceiling(length(ga_dist1)*0.025)]))])
        result_mat[last.num+n+nim.idx,3]<- as.character(names(which.max(table(dist_y_case[as.numeric(names(ga_dist1[1:ceiling(length(ga_dist1)*0.025)]))]))))
        result_mat[last.num+n+nim.idx,11]<- 0.5
        print( (proc.time() - gau.time)[3] )
        
        dis.time <- proc.time()
        # Euclidean 거리 
        eucd_dist <- philentropy::distance(final_dum_dat1, method = "euclidean")
        colnames(eucd_dist) <- 1:dim(eucd_dist)[2]
        eucd_dist1 <- sort(eucd_dist[dim(eucd_dist)[1], -dim(eucd_dist)[2]],decreasing = F)
        #table(dist_y_case[as.numeric(names(eucd_dist1[1:ceiling(length(eucd_dist1)*0.025)]))])
        
        result_mat[last.num+n+nim.idx,4]<- as.character(names(which.max(table(dist_y_case[as.numeric(names(eucd_dist1[1:ceiling(length(eucd_dist1)*0.025)]))]))))
        result_mat[last.num+n+nim.idx,12]<- 0.5
        
        
        # jensen_difference 거리 
        JD_dist <- philentropy::distance(final_dum_dat1, method = "jensen_difference")
        colnames(JD_dist) <- 1:dim(JD_dist)[2]
        JD_dist1 <- sort(JD_dist[dim(JD_dist)[1], -dim(JD_dist)[2]],decreasing = F)
        #table(dist_y_case[as.numeric(names(JD_dist1[1:ceiling(length(JD_dist1)*0.025)]))])
        result_mat[last.num+n+nim.idx,5]<- as.character(names(which.max(table(dist_y_case[as.numeric(names(JD_dist1[1:ceiling(length(JD_dist1)*0.025)]))]))))
        result_mat[last.num+n+nim.idx,13]<- 0.5
        
        # Jaccard 거리 
        jacc_dist <- philentropy::distance(final_dum_dat1, method = "jaccard")
        colnames(jacc_dist) <- 1:dim(jacc_dist)[2]
        jacc_dist1 <- sort(jacc_dist[dim(jacc_dist)[1], -dim(jacc_dist)[2]],decreasing = F)
        #table(dist_y_case[as.numeric(names(jacc_dist1[1:ceiling(length(jacc_dist1)*0.025)]))])
        result_mat[last.num+n+nim.idx,6]<- as.character(names(which.max(table(dist_y_case[as.numeric(names(jacc_dist1[1:ceiling(length(jacc_dist1)*0.025)]))]))))
        result_mat[last.num+n+nim.idx,14]<- 0.5

      }
      
      }
    tot.time <-(proc.time() - NIV.time)[3]
    print(paste0("NIV_",tot.time))
    
  }  
  

  warnings()
  
  # 개수 만큼 더하기
  # 위에서 na 개수 더하기
  last.num <- last.num + length(na.list)
  print(paste("i_",i));print(paste("n_",n))
  print( (proc.time() - ans.time)[3] )
   
}

print( (proc.time() - st.time)[3] )

result_mat <- result_mat[result_mat[1:dim(result_mat)[1] ,1] != 0, ]
result_mat <- result_mat[order(as.numeric(result_mat[,1]),result_mat[,2]),]
write.csv(result_mat,"ML_result_mat.csv",row.names = F)

print(last.num)

