/*
Navicat MySQL Data Transfer

Source Server         : localhost_3306
Source Server Version : 50727
Source Host           : localhost:3306
Source Database       : testdata666

Target Server Type    : MYSQL
Target Server Version : 50727
File Encoding         : 65001

Date: 2019-09-04 17:28:41
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for `distancevalue`
-- ----------------------------
DROP TABLE IF EXISTS `distancevalue`;
CREATE TABLE `distancevalue` (
  `methodId` int(11) NOT NULL,
  `methodName` varchar(1000) DEFAULT NULL,
  `methodParameters` varchar(1000) DEFAULT NULL,
  `methodOfClass` varchar(1000) DEFAULT NULL,
  `className` varchar(45) DEFAULT NULL,
  `distance` varchar(1000) DEFAULT NULL,
  PRIMARY KEY (`methodId`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of distancevalue1
-- ----------------------------
