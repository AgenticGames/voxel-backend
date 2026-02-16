use crate::octree::node::OctreeNode;

/// Depth-first traversal iterator
pub struct OctreeIter<'a> {
    stack: Vec<&'a OctreeNode>,
}

impl<'a> OctreeIter<'a> {
    pub fn new(root: &'a OctreeNode) -> Self {
        Self { stack: vec![root] }
    }
}

impl<'a> Iterator for OctreeIter<'a> {
    type Item = &'a OctreeNode;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.stack.pop()?;
        if let OctreeNode::Branch { children, .. } = node {
            for child in children.iter().rev() {
                self.stack.push(child);
            }
        }
        Some(node)
    }
}
