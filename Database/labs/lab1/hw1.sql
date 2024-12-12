/*
 Navicat Premium Dump SQL

 Source Server         : localhost_3306
 Source Server Type    : MySQL
 Source Server Version : 80039 (8.0.39)
 Source Host           : localhost:3306
 Source Schema         : hw1

 Target Server Type    : MySQL
 Target Server Version : 80039 (8.0.39)
 File Encoding         : 65001

 Date: 21/11/2024 10:49:08
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for account
-- ----------------------------
DROP TABLE IF EXISTS `account`;
CREATE TABLE `account`  (
  `AccountID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `AccountBalance` float NOT NULL,
  `BranchName` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `LatestAccess` date NOT NULL,
  PRIMARY KEY (`AccountID`) USING BTREE,
  INDEX `FK_ACCOUNT_BRANCH`(`BranchName` ASC) USING BTREE,
  CONSTRAINT `FK_ACC_CHECKACC` FOREIGN KEY (`AccountID`) REFERENCES `checkaccout` (`AccountID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_ACC_DEPOACC` FOREIGN KEY (`AccountID`) REFERENCES `depositaccount` (`AccountID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_ACC_OWNACC` FOREIGN KEY (`AccountID`) REFERENCES `ownaccount` (`AccountID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_ACCOUNT_BRANCH` FOREIGN KEY (`BranchName`) REFERENCES `bankbranch` (`BranchName`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of account
-- ----------------------------

-- ----------------------------
-- Table structure for bankbranch
-- ----------------------------
DROP TABLE IF EXISTS `bankbranch`;
CREATE TABLE `bankbranch`  (
  `BranchName` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `BranchCity` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `Capacity` float NOT NULL,
  PRIMARY KEY (`BranchName`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of bankbranch
-- ----------------------------

-- ----------------------------
-- Table structure for checkaccout
-- ----------------------------
DROP TABLE IF EXISTS `checkaccout`;
CREATE TABLE `checkaccout`  (
  `AccountID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `Overdraft` float NOT NULL,
  PRIMARY KEY (`AccountID`) USING BTREE,
  CONSTRAINT `FK_CHECKAC_AC` FOREIGN KEY (`AccountID`) REFERENCES `account` (`AccountID`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of checkaccout
-- ----------------------------

-- ----------------------------
-- Table structure for commonstaff
-- ----------------------------
DROP TABLE IF EXISTS `commonstaff`;
CREATE TABLE `commonstaff`  (
  `StaffID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `LeaderID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`StaffID`) USING BTREE,
  INDEX `LeaderID`(`LeaderID` ASC) USING BTREE,
  CONSTRAINT `FK_COMMON_LEADER` FOREIGN KEY (`LeaderID`) REFERENCES `manager` (`StaffID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_STAFF_COMMONSTAFF` FOREIGN KEY (`StaffID`) REFERENCES `staff` (`StaffID`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of commonstaff
-- ----------------------------

-- ----------------------------
-- Table structure for customer
-- ----------------------------
DROP TABLE IF EXISTS `customer`;
CREATE TABLE `customer`  (
  `CustomerName` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `CustomerID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `CustomerStreet` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `CustomerCity` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`CustomerID`) USING BTREE,
  INDEX `CustomerID`(`CustomerID` ASC) USING BTREE,
  CONSTRAINT `FK_CUST_ACC` FOREIGN KEY (`CustomerID`) REFERENCES `ownaccount` (`CustomerID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_CUST_CUSTLOAN` FOREIGN KEY (`CustomerID`) REFERENCES `customerloan` (`CustomerID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_CUST_STAFFRES` FOREIGN KEY (`CustomerID`) REFERENCES `staffresponsible` (`CustomerID`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of customer
-- ----------------------------

-- ----------------------------
-- Table structure for customerloan
-- ----------------------------
DROP TABLE IF EXISTS `customerloan`;
CREATE TABLE `customerloan`  (
  `LoanID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `CustomerID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`LoanID`, `CustomerID`) USING BTREE,
  INDEX `FK_LOAN_CUSTOMER`(`CustomerID` ASC) USING BTREE,
  INDEX `LoanID`(`LoanID` ASC) USING BTREE,
  CONSTRAINT `FK_LOAN_CUSTOMER` FOREIGN KEY (`CustomerID`) REFERENCES `customer` (`CustomerID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_LOANID` FOREIGN KEY (`LoanID`) REFERENCES `loans` (`LoanID`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of customerloan
-- ----------------------------

-- ----------------------------
-- Table structure for depositaccount
-- ----------------------------
DROP TABLE IF EXISTS `depositaccount`;
CREATE TABLE `depositaccount`  (
  `AccountID` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `Interest` float NOT NULL,
  PRIMARY KEY (`AccountID`) USING BTREE,
  CONSTRAINT `FK_DEPOSITACC_ACC` FOREIGN KEY (`AccountID`) REFERENCES `account` (`AccountID`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of depositaccount
-- ----------------------------

-- ----------------------------
-- Table structure for loans
-- ----------------------------
DROP TABLE IF EXISTS `loans`;
CREATE TABLE `loans`  (
  `LoanID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `BranchName` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `Amount` float NULL DEFAULT NULL,
  `Times` int NULL DEFAULT NULL,
  PRIMARY KEY (`LoanID`) USING BTREE,
  INDEX `FK_BRANCH_LOAN`(`BranchName` ASC) USING BTREE,
  CONSTRAINT `FK_BRANCH_LOAN` FOREIGN KEY (`BranchName`) REFERENCES `bankbranch` (`BranchName`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_LOAN_CUSTLOAN` FOREIGN KEY (`LoanID`) REFERENCES `customerloan` (`LoanID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_LOAN_PAY` FOREIGN KEY (`LoanID`) REFERENCES `payment` (`LoanID`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of loans
-- ----------------------------

-- ----------------------------
-- Table structure for manager
-- ----------------------------
DROP TABLE IF EXISTS `manager`;
CREATE TABLE `manager`  (
  `StaffID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `Department` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`StaffID`) USING BTREE,
  CONSTRAINT `FK_MANAGER_COMMONSTAFF` FOREIGN KEY (`StaffID`) REFERENCES `commonstaff` (`LeaderID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_MANAGER_STAFF` FOREIGN KEY (`StaffID`) REFERENCES `staff` (`StaffID`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of manager
-- ----------------------------

-- ----------------------------
-- Table structure for ownaccount
-- ----------------------------
DROP TABLE IF EXISTS `ownaccount`;
CREATE TABLE `ownaccount`  (
  `CustomerID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `AccountID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`CustomerID`, `AccountID`) USING BTREE,
  INDEX `FK_OWNACC_ACCID`(`AccountID` ASC) USING BTREE,
  INDEX `CustomerID`(`CustomerID` ASC) USING BTREE,
  CONSTRAINT `FK_OWNACC_ACCID` FOREIGN KEY (`AccountID`) REFERENCES `account` (`AccountID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_OWNACC_CUSTOMERID` FOREIGN KEY (`CustomerID`) REFERENCES `customer` (`CustomerID`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of ownaccount
-- ----------------------------

-- ----------------------------
-- Table structure for payment
-- ----------------------------
DROP TABLE IF EXISTS `payment`;
CREATE TABLE `payment`  (
  `LoanID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `PayID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `PayAmount` float NULL DEFAULT NULL,
  `PayDate` date NULL DEFAULT NULL,
  PRIMARY KEY (`LoanID`, `PayID`) USING BTREE,
  INDEX `LoanID`(`LoanID` ASC) USING BTREE,
  CONSTRAINT `FK_PAY_LOANID` FOREIGN KEY (`LoanID`) REFERENCES `loans` (`LoanID`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of payment
-- ----------------------------

-- ----------------------------
-- Table structure for staff
-- ----------------------------
DROP TABLE IF EXISTS `staff`;
CREATE TABLE `staff`  (
  `StaffName` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `StaffID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `StaffTel` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `StaffAddress` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `StaffEnrollDate` date NOT NULL,
  `BranchName` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`StaffID` DESC) USING BTREE,
  INDEX `FK_EMPLOY_BRANCH`(`BranchName` ASC) USING BTREE,
  INDEX `FK_STAFF_STAFFRES`(`StaffID` ASC) USING BTREE,
  CONSTRAINT `FK_EMPLOY_BRANCH` FOREIGN KEY (`BranchName`) REFERENCES `bankbranch` (`BranchName`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_STAFF_COMSTAFF` FOREIGN KEY (`StaffID`) REFERENCES `commonstaff` (`StaffID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_STAFF_MANAGE` FOREIGN KEY (`StaffID`) REFERENCES `manager` (`StaffID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_STAFF_STAFFRES` FOREIGN KEY (`StaffID`) REFERENCES `staffresponsible` (`StaffID`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of staff
-- ----------------------------

-- ----------------------------
-- Table structure for staffresponsible
-- ----------------------------
DROP TABLE IF EXISTS `staffresponsible`;
CREATE TABLE `staffresponsible`  (
  `StaffID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `CustomerID` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `ResType` int NULL DEFAULT NULL,
  PRIMARY KEY (`StaffID` DESC, `CustomerID`) USING BTREE,
  INDEX `FK_RES_CUSTOMER`(`CustomerID` ASC) USING BTREE,
  INDEX `FK_RES_STAFF`(`StaffID` ASC) USING BTREE,
  CONSTRAINT `FK_RES_CUSTOMER` FOREIGN KEY (`CustomerID`) REFERENCES `customer` (`CustomerID`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `FK_RES_STAFF` FOREIGN KEY (`StaffID`) REFERENCES `staff` (`StaffID`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of staffresponsible
-- ----------------------------

SET FOREIGN_KEY_CHECKS = 1;
