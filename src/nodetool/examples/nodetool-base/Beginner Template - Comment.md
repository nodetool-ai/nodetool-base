# 🎓 Beginner Template: Creative Story Ideas Generator

This workflow demonstrates the core concepts of NodeTool:

## 📥 INPUTS (Green nodes on left)

• Three StringInput nodes let you customize the story
• Try changing Genre, Character, or Setting values

## 📝 TEMPLATE FORMATTING

• The constant template defines the prompt structure
• FormatText node combines inputs with {{PLACEHOLDERS}}
• Dynamic properties map inputs to template variables

## 🤖 AI GENERATION

• ListGenerator streams multiple story ideas
• Each idea appears one at a time (iteration/streaming)
• Downstream nodes process each item automatically

## 👁️ OUTPUT & PREVIEW

• Preview nodes display intermediate and final results
• Top preview shows the formatted prompt
• Bottom preview shows all generated story ideas

## 🎯 KEY LEARNING POINTS:

1. Data flows left-to-right through connected nodes
2. Edges connect outputs (right) to inputs (left)
3. Templates use {{VARIABLES}} for dynamic content
4. Streaming nodes emit multiple values over time
5. Preview nodes help debug and visualize data

## 💡 TRY THIS:

• Click the input nodes and change their values
• Run the workflow and watch results appear
• Modify the template to add more instructions
• Try connecting nodes in different ways