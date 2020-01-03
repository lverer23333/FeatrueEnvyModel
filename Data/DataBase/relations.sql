/*
Navicat MySQL Data Transfer

Source Server         : localhost_3306
Source Server Version : 50727
Source Host           : localhost:3306
Source Database       : testdata666

Target Server Type    : MYSQL
Target Server Version : 50727
File Encoding         : 65001

Date: 2019-09-04 17:29:02
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for `relations`
-- ----------------------------
DROP TABLE IF EXISTS `relations`;
CREATE TABLE `relations` (
  `KeyNum` int(11) NOT NULL,
  `ClassID` int(11) DEFAULT NULL,
  `MethodID` int(11) DEFAULT NULL,
  `MethodInThisClassOrNot` int(11) DEFAULT NULL,
  PRIMARY KEY (`KeyNum`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of relations
-- ----------------------------
