/*
Navicat MySQL Data Transfer

Source Server         : localhost_3306
Source Server Version : 50727
Source Host           : localhost:3306
Source Database       : testdata666

Target Server Type    : MYSQL
Target Server Version : 50727
File Encoding         : 65001

Date: 2019-09-04 17:28:52
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for `methodinfo`
-- ----------------------------
DROP TABLE IF EXISTS `methodinfo`;
CREATE TABLE `methodinfo` (
  `MethodID` int(11) NOT NULL,
  `MethodName` varchar(1000) DEFAULT NULL,
  `MethodParameters` varchar(1000) DEFAULT NULL,
  `MethodOfClass` varchar(1000) DEFAULT NULL,
  PRIMARY KEY (`MethodID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of methodinfo
-- ----------------------------
