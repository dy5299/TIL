# TIL (Today I Learned)

> 매일 배운 내용을 정리합니다.

## 1. git

* [git 기초](./git.md)
* [마크다운 활용](./markdown.md)

```bash
$ git status
On branch master
Your branch is up to date with 'origin/master'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        README.md

nothing added to commit but untracked files present (use "git add" to track)

$ git add .

$ git commit -m 'Add README.md'
[master 87ae314] Add README.md
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 README.md

$ git push origin master
Enumerating objects: 4, done.
Counting objects: 100% (4/4), done.
Delta compression using up to 12 threads
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 340 bytes | 340.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To https://github.com/dy5299/TIL.git
   549744b..87ae314  master -> master
```

## 2. Python



