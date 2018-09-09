
library("Hmisc")
setwd("C:/Users/korea/Desktop/dataset_kor")

# ML 결과 파일

ML_result_mat<-as.matrix(read.csv("ML_result_mat.csv",sep=','))
ML_result_mat <- ML_result_mat[order(as.numeric(ML_result_mat[,1]),ML_result_mat[,2]),]

# 답안지 불러오기
result_kor <- as.matrix(read.csv("result_kor.csv",sep=','))


# DL 결과 파일
DL_result_mat<-as.matrix(read.csv("DL_result_mat.csv",sep=','))
DL_result_mat <- DL_result_mat[order(as.numeric(DL_result_mat[,1]),DL_result_mat[,2]),]

# Boost 결과 파일
BS_result_mat<-as.matrix(read.csv("BS_result_mat.csv",sep=','))
BS_result_mat <- BS_result_mat[order(as.numeric(BS_result_mat[,1]),BS_result_mat[,2]),]

BS_result_mat2<-as.matrix(read.csv("BS_result_mat2.csv",sep=','))
BS_result_mat2 <- BS_result_mat2[order(as.numeric(BS_result_mat2[,1]),BS_result_mat2[,2]),]

# 원본 
test_kor<-as.matrix(read.csv("test_kor.csv",sep=','))
test_kor[test_kor==''] <- NA

test_col <- c("주야",	"요일",	"사망자수",	"사상자수",	"중상자수",	"경상자수",	"부상신고자수",	"발생지시도",	
              "발생지시군구",	"사고유형_대분류",	"사고유형_중분류",	"법규위반",	"도로형태_대분류",	"도로형태",	"당사자종별_1당_대분류",	"당사자종별_2당_대분류")

cat_list1 <- c("주야", "요일", "사고유형_대분류","사고유형_중분류", "법규위반", "도로형태_대분류", "도로형태",
               "당사자종별_1당_대분류", "당사자종별_2당_대분류","발생지시도")
num_list1 <- c( "사망자수","중상자수","경상자수","부상신고자수")
num_list2 <- c( "사망자수","중상자수","경상자수","부상신고자수","사상자수")

colnames(test_kor) <- test_col


last.num <-0
test_kor1 <- test_kor
for(i in 1:dim(test_kor)[1]){  # dim(test_kor)[1]
  test_obs <- test_kor[i,] 
  na.list <- which(is.na(test_obs))
  # 빈 변수 채우기
  for(n.idx in 1:length(na.list))
  {
    #n.idx <-2
    # 수치 결과 채우기 / 사상자 제외
    if(names(test_obs)[na.list][n.idx] %in% num_list1 )
    {
      result1 <- as.numeric(ML_result_mat[last.num + n.idx , 3:10 * (as.numeric(ML_result_mat[last.num+n.idx,11:18]) != 0) ])
      result2 <- as.numeric(BS_result_mat[last.num + n.idx , 3])
      result3 <- as.numeric(BS_result_mat2[last.num + n.idx , 3])
      result4 <- as.numeric(DL_result_mat[last.num + n.idx , 3])
      result <- c(result1,result2,result3,result4)
      
      weight1 <- as.numeric(ML_result_mat[last.num + n.idx , 11:18 * (as.numeric(ML_result_mat[last.num+n.idx,11:18]) != 0) ])
      weight2 <- as.numeric(BS_result_mat[last.num + n.idx , 4])
      weight3 <- as.numeric(BS_result_mat2[last.num + n.idx , 4])
      weight4 <- as.numeric(DL_result_mat[last.num + n.idx , 4])
      weight <- c(weight1,weight2,weight3,weight4)
      
      weight <- weight/sum(weight)
      
      
      result_kor[last.num + n.idx,3] <- sum(result*weight)
      test_kor1[i, match(ML_result_mat[last.num + n.idx,2],LETTERS)] <- sum(result*weight)
    }
    # 범주 결과 채우기
    if(names(test_obs)[na.list][n.idx] %nin% num_list2 )
    {
      
      result1 <- as.character(ML_result_mat[last.num + n.idx , 3:10 * (as.numeric(ML_result_mat[last.num+n.idx,11:18]) != 0) ])
      result2 <- as.character(BS_result_mat[last.num + n.idx , 3])
      result3 <- as.character(BS_result_mat2[last.num + n.idx , 3])
      result4 <- as.character(DL_result_mat[last.num + n.idx , 3])
      result <- c(result1,result2,result3,result4)
      
      weight1 <- as.numeric(ML_result_mat[last.num + n.idx , 11:18 * (as.numeric(ML_result_mat[last.num+n.idx,11:18]) != 0) ])
      weight2 <-  as.numeric(BS_result_mat[last.num + n.idx , 4])
      weight3 <-  as.numeric(BS_result_mat2[last.num + n.idx , 4])
      weight4 <-as.numeric(DL_result_mat[last.num + n.idx , 4])
      weight <- c(weight1,weight2,weight3,weight4)
      
      weight <- weight/sum(weight)
      
      
      uni.rslt <- unique(result)
      wt.vote <- matrix(0, nrow =  1,ncol=length(uni.rslt))
      colnames(wt.vote) <- uni.rslt 
      
      for(uni.idx in 1:length(uni.rslt))
      {
        wt.vote[1,uni.idx] <- sum(weight[result == uni.rslt[uni.idx]])
        
      }
      #View(test_kor1)
      
      result_kor[last.num + n.idx,3] <-  colnames(wt.vote)[which.max(wt.vote)]
      test_kor1[i, match(ML_result_mat[last.num + n.idx,2],LETTERS)] <- colnames(wt.vote)[which.max(wt.vote)]
     
    }
  }
  
  last.num <- last.num + sum(is.na(test_obs))
}
warnings()
## 사상자 
last.num <-0
for(i in 1:dim(test_kor)[1]){  # dim(test_kor)[1]
  test_obs <- test_kor1[i,] # i<-1
  test_obs1 <- test_kor[i,] 
  na.list <- which(is.na(test_obs1))
  
  for(n.idx in 1:length(na.list))
  {
    # n.idx<-2
  if(names(test_obs)[na.list][n.idx] %in% "사상자수"  )
  {
    
    result_kor[last.num + n.idx,3] <-  sum(as.numeric(test_obs[names(test_obs) %in% num_list1]))
    test_kor1[i, match(ML_result_mat[last.num + n.idx,2],LETTERS)] <- sum(as.numeric(test_obs[names(test_obs) %in% num_list1]))
  } 
  }
  last.num <- last.num + sum(is.na(test_obs1))
}
View(result_kor)
View(test_kor1)

result_kor <- result_kor[result_kor[1:dim(result_kor)[1] ,1] != 0, ]
result_kor <- result_kor[order(as.numeric(result_kor[,1]),result_kor[,2]),]
colnames(result_kor) <- c( "행","열","값")
write.csv(result_kor,"result.csv",row.names = F)

